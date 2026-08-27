package detorrent

import (
	"math/rand"
	"testing"
	"time"
)

func TestArtifactBinBoundaries(t *testing.T) {
	boundaries := artifactBinBoundaries()
	if len(boundaries) != 257 {
		t.Fatalf("got %d boundaries, want 257", len(boundaries))
	}
	if boundaries[0] != 0 {
		t.Fatalf("first boundary is %v, want zero", boundaries[0])
	}
	if difference := boundaries[len(boundaries)-1] - 49*time.Second; difference < -time.Microsecond || difference > time.Microsecond {
		t.Fatalf("last boundary is %v, want 49s", boundaries[len(boundaries)-1])
	}
	for index := 1; index < len(boundaries); index++ {
		if boundaries[index] <= boundaries[index-1] {
			t.Fatalf("boundaries are not increasing at %d", index)
		}
	}
}

func TestExponentialSchedule(t *testing.T) {
	start := time.Unix(100, 0)
	width := 250 * time.Millisecond
	timestamps := exponentialSchedule(start, width, 50, rand.New(rand.NewSource(2024)))
	if len(timestamps) != 50 {
		t.Fatalf("got %d timestamps, want 50", len(timestamps))
	}
	for index, timestamp := range timestamps {
		if !timestamp.After(start) || !timestamp.Before(start.Add(width)) {
			t.Fatalf("timestamp %d falls outside the bin: %v", index, timestamp)
		}
		if index > 0 && timestamp.Before(timestamps[index-1]) {
			t.Fatalf("timestamps are not sorted at %d", index)
		}
	}
}
