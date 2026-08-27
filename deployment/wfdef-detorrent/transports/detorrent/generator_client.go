package detorrent

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"
)

type generatorClient struct {
	conn    net.Conn
	reader  *bufio.Reader
	encoder *json.Encoder
	mutex   sync.Mutex
}

type generatorResponse struct {
	SessionID    string  `json:"session_id"`
	Bin          int     `json:"bin"`
	Dummy        int     `json:"dummy_packets"`
	Closed       bool    `json:"closed"`
	Error        string  `json:"error"`
	MeanRawTotal float64 `json:"mean_raw_total"`
}

func dialGenerator(address string, timeout time.Duration) (*generatorClient, error) {
	conn, err := net.DialTimeout("tcp", address, timeout)
	if err != nil {
		return nil, err
	}
	return &generatorClient{
		conn:    conn,
		reader:  bufio.NewReader(conn),
		encoder: json.NewEncoder(conn),
	}, nil
}

func (c *generatorClient) request(request interface{}) (generatorResponse, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if err := c.encoder.Encode(request); err != nil {
		return generatorResponse{}, err
	}
	line, err := c.reader.ReadBytes('\n')
	if err != nil {
		return generatorResponse{}, err
	}
	var response generatorResponse
	if err := json.Unmarshal(line, &response); err != nil {
		return generatorResponse{}, err
	}
	if response.Error != "" {
		return generatorResponse{}, fmt.Errorf("generator: %s", response.Error)
	}
	return response, nil
}

func (c *generatorClient) start(budget int) (string, int, error) {
	response, err := c.request(map[string]interface{}{
		"op":     "start",
		"budget": budget,
	})
	return response.SessionID, response.Dummy, err
}

func (c *generatorClient) step(sessionID string, previousBin, realPackets int) (int, error) {
	response, err := c.request(map[string]interface{}{
		"op":                    "step",
		"session_id":            sessionID,
		"previous_bin":          previousBin,
		"real_download_packets": realPackets,
	})
	return response.Dummy, err
}

func (c *generatorClient) closeSession(sessionID string) error {
	_, err := c.request(map[string]interface{}{
		"op":         "close",
		"session_id": sessionID,
	})
	return err
}

func (c *generatorClient) Close() error {
	return c.conn.Close()
}
