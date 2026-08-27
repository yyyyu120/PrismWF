package detorrent

import (
	"io"
	"math"
	"math/rand"
	"net"
	"sync"
	"sync/atomic"
	"time"

	pt "git.torproject.org/pluggable-transports/goptlib.git"
	"github.com/websitefingerprinting/wfdef.git/common/utils"
	"github.com/websitefingerprinting/wfdef.git/transports/base"
	"github.com/websitefingerprinting/wfdef.git/transports/defconn"
)

const (
	transportName       = "detorrent"
	budgetArg           = "budget"
	generatorAddrArg    = "generator-addr"
	defaultGeneratorAddr = "127.0.0.1:19991"
	startPacketCount    = 10
	introMean           = 100 * time.Millisecond
	uploadTick          = 2 * time.Millisecond
	uploadRatio         = 5.0
)

type Transport struct{ defconn.Transport }

func (transport *Transport) Name() string { return transportName }

type clientArgs struct {
	*defconn.DefConnClientArgs
	budget        int
	generatorAddr string
}

type clientFactory struct{ *defconn.DefConnClientFactory }

func (transport *Transport) ClientFactory(stateDir string) (base.ClientFactory, error) {
	factory, err := transport.Transport.ClientFactory(stateDir)
	return &clientFactory{factory.(*defconn.DefConnClientFactory)}, err
}

func (factory *clientFactory) Transport() base.Transport { return factory.DefConnClientFactory.Transport() }

func (factory *clientFactory) ParseArgs(args *pt.Args) (interface{}, error) {
	baseArgs, err := factory.DefConnClientFactory.ParseArgs(args)
	if err != nil {
		return nil, err
	}
	budget, err := utils.GetIntArgFromStr(budgetArg, args)
	if err != nil {
		return nil, err
	}
	generatorAddr, ok := args.Get(generatorAddrArg)
	if !ok || generatorAddr == "" {
		generatorAddr = defaultGeneratorAddr
	}
	return &clientArgs{baseArgs.(*defconn.DefConnClientArgs), budget.(int), generatorAddr}, nil
}

func (factory *clientFactory) Dial(network, address string, dialFn base.DialFunc, args interface{}) (net.Conn, error) {
	baseConn, err := factory.DefConnClientFactory.Dial(network, address, dialFn, args)
	if err != nil {
		return nil, err
	}
	parsed := args.(*clientArgs)
	return newConn(baseConn.(*defconn.DefConn), parsed.budget, parsed.generatorAddr), nil
}

type serverFactory struct {
	*defconn.DefConnServerFactory
	budget        int
	generatorAddr string
}

func (transport *Transport) ServerFactory(stateDir string, args *pt.Args) (base.ServerFactory, error) {
	baseFactory, err := transport.Transport.ServerFactory(stateDir, args)
	if err != nil {
		return nil, err
	}
	state, err := serverStateFromArgs(stateDir, args)
	if err != nil {
		return nil, err
	}
	return &serverFactory{baseFactory.(*defconn.DefConnServerFactory), state.budget, state.generatorAddr}, nil
}

func (factory *serverFactory) WrapConn(raw net.Conn) (net.Conn, error) {
	baseConn, err := factory.DefConnServerFactory.WrapConn(raw)
	if err != nil {
		return nil, err
	}
	return newConn(baseConn.(*defconn.DefConn), factory.budget, factory.generatorAddr), nil
}

type detorrentConn struct {
	*defconn.DefConn
	budget        int
	generatorAddr string
	boundaries    []time.Duration
	rng           *rand.Rand
	rngMutex      sync.Mutex

	mutex             sync.Mutex
	introActive       bool
	introGeneration   uint64
	defenseStart      time.Time
	defenseGeneration uint64
	sessionID         string
	realDownloadBins  [numBins]int
	downloadCount     int64
	uploadCount       int64
	realPacketCount   int64
	lastDownload      time.Time
	lastUpload        time.Time
	downloadRate      float64
	uploadRate        float64
}

func newConn(baseConn *defconn.DefConn, budget int, generatorAddr string) *detorrentConn {
	return &detorrentConn{
		DefConn:       baseConn,
		budget:        budget,
		generatorAddr: generatorAddr,
		boundaries:    artifactBinBoundaries(),
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func (conn *detorrentConn) startDefense() {
	conn.mutex.Lock()
	if !conn.defenseStart.IsZero() {
		conn.mutex.Unlock()
		return
	}
	conn.defenseStart = time.Now()
	conn.defenseGeneration++
	generation := conn.defenseGeneration
	conn.realDownloadBins = [numBins]int{}
	conn.mutex.Unlock()
	if conn.IsServer {
		go conn.runDownloadDefense(generation)
	}
}

func (conn *detorrentConn) stopDefense() {
	conn.mutex.Lock()
	conn.defenseStart = time.Time{}
	conn.defenseGeneration++
	conn.mutex.Unlock()
}

func (conn *detorrentConn) beginIntro() {
	conn.mutex.Lock()
	if conn.introActive || !conn.defenseStart.IsZero() {
		conn.mutex.Unlock()
		return
	}
	conn.introActive = true
	conn.introGeneration++
	generation := conn.introGeneration
	conn.mutex.Unlock()
	go conn.runIntroDefense(generation)
}

func (conn *detorrentConn) stopIntro() {
	conn.mutex.Lock()
	conn.introActive = false
	conn.introGeneration++
	conn.mutex.Unlock()
}

func (conn *detorrentConn) runIntroDefense(generation uint64) {
	for {
		conn.rngMutex.Lock()
		delay := time.Duration(conn.rng.ExpFloat64() * float64(introMean))
		conn.rngMutex.Unlock()
		if !sleepFor(delay, conn.CloseChan) || !conn.introGenerationActive(generation) {
			return
		}
		conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeDummy, PadLen: defconn.MaxPacketPaddingLength}
		if conn.IsServer {
			conn.recordDownload(false)
		} else {
			conn.recordUpload()
		}
	}
}

func (conn *detorrentConn) introGenerationActive(generation uint64) bool {
	conn.mutex.Lock()
	defer conn.mutex.Unlock()
	return conn.introActive && conn.introGeneration == generation
}

func (conn *detorrentConn) recordDownload(real bool) {
	atomic.AddInt64(&conn.downloadCount, 1)
	conn.mutex.Lock()
	now := time.Now()
	conn.downloadRate = updateRate(conn.downloadRate, conn.lastDownload, now)
	conn.lastDownload = now
	if real && !conn.defenseStart.IsZero() {
		elapsed := now.Sub(conn.defenseStart)
		for bin := 0; bin < numBins; bin++ {
			if elapsed >= conn.boundaries[bin] && elapsed < conn.boundaries[bin+1] {
				conn.realDownloadBins[bin]++
				break
			}
		}
	}
	conn.mutex.Unlock()
}

func (conn *detorrentConn) recordUpload() {
	atomic.AddInt64(&conn.uploadCount, 1)
	conn.mutex.Lock()
	now := time.Now()
	conn.uploadRate = updateRate(conn.uploadRate, conn.lastUpload, now)
	conn.lastUpload = now
	conn.mutex.Unlock()
}

func updateRate(rate float64, previous, now time.Time) float64 {
	if previous.IsZero() {
		return 1
	}
	return (rate + 1) * math.Exp(-now.Sub(previous).Seconds())
}

func currentRate(rate float64, previous, now time.Time) float64 {
	if previous.IsZero() {
		return 1
	}
	return rate * math.Exp(-now.Sub(previous).Seconds())
}

func (conn *detorrentConn) recordRealPacketAndMaybeStart() {
	// budget=0 is the cell-framed Null control: preserve DefConn framing while
	// disabling both the introductory padding and the generated padding cycle.
	if conn.IsServer || conn.budget == 0 {
		return
	}
	count := atomic.AddInt64(&conn.realPacketCount, 1)
	if count == 1 {
		conn.beginIntro()
		conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeSignalStart, PadLen: defconn.MaxPacketPaddingLength}
	}
	if count == startPacketCount {
		conn.stopIntro()
		conn.startDefense()
		conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeFinish, PadLen: defconn.MaxPacketPaddingLength}
	}
}

func (conn *detorrentConn) runDownloadDefense(generation uint64) {
	client, err := dialGenerator(conn.generatorAddr, 5*time.Second)
	if err != nil {
		conn.ErrChan <- err
		return
	}
	defer client.Close()

	sessionID, dummyCount, err := client.start(conn.budget)
	if err != nil {
		conn.ErrChan <- err
		return
	}
	defer client.closeSession(sessionID)

	conn.mutex.Lock()
	conn.sessionID = sessionID
	start := conn.defenseStart
	conn.mutex.Unlock()

	for bin := 0; bin < numBins; bin++ {
		if !conn.generationActive(generation) {
			return
		}
		binStart := start.Add(conn.boundaries[bin])
		binEnd := start.Add(conn.boundaries[bin+1])
		conn.rngMutex.Lock()
		timestamps := exponentialSchedule(binStart, binEnd.Sub(binStart), dummyCount, conn.rng)
		conn.rngMutex.Unlock()
		for _, timestamp := range timestamps {
			if !sleepUntil(timestamp, conn.CloseChan) || !conn.generationActive(generation) {
				return
			}
			conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeDummy, PadLen: defconn.MaxPacketPaddingLength}
			conn.recordDownload(false)
		}
		if !sleepUntil(binEnd, conn.CloseChan) {
			return
		}
		if bin == numBins-1 {
			break
		}
		conn.mutex.Lock()
		realPackets := conn.realDownloadBins[bin]
		conn.mutex.Unlock()
		dummyCount, err = client.step(sessionID, bin, realPackets)
		if err != nil {
			conn.ErrChan <- err
			return
		}
	}
	conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeFinish, PadLen: defconn.MaxPacketPaddingLength}
	conn.stopDefense()
}

func (conn *detorrentConn) generationActive(generation uint64) bool {
	conn.mutex.Lock()
	defer conn.mutex.Unlock()
	return !conn.defenseStart.IsZero() && conn.defenseGeneration == generation
}

func sleepUntil(deadline time.Time, closed <-chan int) bool {
	delay := time.Until(deadline)
	if delay <= 0 {
		return true
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-closed:
		return false
	}
}

func sleepFor(delay time.Duration, closed <-chan int) bool {
	return sleepUntil(time.Now().Add(delay), closed)
}

func (conn *detorrentConn) runUploadDefense() {
	ticker := time.NewTicker(uploadTick)
	defer ticker.Stop()
	for {
		select {
		case <-conn.CloseChan:
			return
		case now := <-ticker.C:
			conn.mutex.Lock()
			active := !conn.defenseStart.IsZero()
			downloadRate := currentRate(conn.downloadRate, conn.lastDownload, now)
			uploadRate := currentRate(conn.uploadRate, conn.lastUpload, now)
			conn.mutex.Unlock()
			if !active {
				continue
			}
			targetRate := downloadRate / uploadRatio
			if targetRate < 1 || uploadRate >= targetRate {
				continue
			}
			probability := (targetRate - uploadRate) / 500
			conn.rngMutex.Lock()
			sendDummy := conn.rng.Float64() < probability
			conn.rngMutex.Unlock()
			if sendDummy {
				conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypeDummy, PadLen: defconn.MaxPacketPaddingLength}
				conn.recordUpload()
			}
		}
	}
}

func (conn *detorrentConn) ReadFrom(reader io.Reader) (written int64, err error) {
	defer close(conn.CloseChan)
	go conn.Send()
	if !conn.IsServer {
		go conn.runUploadDefense()
	}

	buffer := make([]byte, 65535)
	for {
		readLength, readError := reader.Read(buffer)
		if readError != nil {
			return written, readError
		}
		for offset := 0; offset < readLength; offset += defconn.MaxPacketPayloadLength {
			end := offset + defconn.MaxPacketPayloadLength
			if end > readLength {
				end = readLength
			}
			payload := append([]byte(nil), buffer[offset:end]...)
			conn.SendChan <- defconn.PacketInfo{PktType: defconn.PacketTypePayload, Data: payload, PadLen: uint16(defconn.MaxPacketPaddingLength-len(payload))}
			written += int64(len(payload))
			if !conn.IsServer {
				conn.recordUpload()
				conn.recordRealPacketAndMaybeStart()
			} else {
				conn.recordDownload(true)
			}
		}
	}
}

func (conn *detorrentConn) Read(buffer []byte) (int, error) {
	return conn.DefConn.MyRead(buffer, conn.readPackets)
}

var _ base.ClientFactory = (*clientFactory)(nil)
var _ base.ServerFactory = (*serverFactory)(nil)
var _ base.Transport = (*Transport)(nil)
var _ net.Conn = (*detorrentConn)(nil)
