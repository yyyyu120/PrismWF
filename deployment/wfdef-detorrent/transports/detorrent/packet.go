package detorrent

import (
	"encoding/binary"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/websitefingerprinting/wfdef.git/common/drbg"
	"github.com/websitefingerprinting/wfdef.git/common/log"
	"github.com/websitefingerprinting/wfdef.git/transports/defconn"
	"github.com/websitefingerprinting/wfdef.git/transports/defconn/framing"
)

func (conn *detorrentConn) readPackets() (err error) {
	readLength, readError := conn.Conn.Read(conn.ReadBuffer)
	conn.ReceiveBuffer.Write(conn.ReadBuffer[:readLength])

	var decoded [framing.MaximumFramePayloadLength]byte
	for conn.ReceiveBuffer.Len() > 0 {
		decodedLength, decodeError := conn.Decoder.Decode(decoded[:], conn.ReceiveBuffer)
		if decodeError == framing.ErrAgain {
			break
		}
		if decodeError != nil {
			err = decodeError
			break
		}
		if decodedLength < defconn.PacketOverhead {
			err = defconn.InvalidPacketLengthError(decodedLength)
			break
		}

		packet := decoded[:decodedLength]
		packetType := packet[0]
		payloadLength := binary.BigEndian.Uint16(packet[1:])
		if int(payloadLength) > len(packet)-defconn.PacketOverhead {
			err = defconn.InvalidPayloadLengthError(int(payloadLength))
			break
		}
		payload := packet[3 : 3+payloadLength]

		if !conn.IsServer && packetType != defconn.PacketTypePrngSeed && defconn.LogEnabled {
			log.Infof("[TRACE_LOG] %d %d %d", time.Now().UnixNano(), -int64(payloadLength), -(int64(decodedLength-defconn.PacketOverhead)-int64(payloadLength)))
		}

		switch packetType {
		case defconn.PacketTypePayload:
			if payloadLength > 0 {
				conn.ReceiveDecodedBuffer.Write(payload)
				if !conn.IsServer {
					conn.recordDownload(true)
					conn.recordRealPacketAndMaybeStart()
				}
			}
		case defconn.PacketTypePrngSeed:
			if len(payload) == defconn.SeedPacketPayloadLength && !conn.IsServer {
				seed, seedError := drbg.SeedFromBytes(payload)
				if seedError != nil {
					err = seedError
					break
				}
				conn.LenDist.Reset(seed)
			}
		case defconn.PacketTypeSignalStart:
			if !conn.IsServer {
				panic(fmt.Sprintf("client received SignalStart"))
			}
			conn.beginIntro()
		case defconn.PacketTypeSignalStop:
			if !conn.IsServer {
				panic(fmt.Sprintf("client received SignalStop"))
			}
			conn.stopDefense()
		case defconn.PacketTypeDummy:
			if !conn.IsServer {
				conn.recordDownload(false)
			}
		case defconn.PacketTypeFinish:
			if conn.IsServer {
				conn.stopIntro()
				conn.startDefense()
			} else {
				conn.stopDefense()
				atomic.StoreInt64(&conn.realPacketCount, 0)
			}
		}
	}

	if readError != nil {
		return readError
	}
	return err
}
