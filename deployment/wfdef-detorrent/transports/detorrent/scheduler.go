package detorrent

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

const numBins = 256

func artifactBinBoundaries() []time.Duration {
	boundaries := make([]time.Duration, numBins+1)
	for index := range boundaries {
		seconds := math.Exp(math.Log(50)*float64(index)/numBins) - 1
		boundaries[index] = time.Duration(seconds * float64(time.Second))
	}
	return boundaries
}

// exponentialSchedule samples exponential gaps conditioned on exactly count
// events falling inside the bin. The resulting timestamps are sorted and do
// not include either bin boundary.
func exponentialSchedule(start time.Time, width time.Duration, count int, rng *rand.Rand) []time.Time {
	if count <= 0 || width <= 0 {
		return nil
	}

	gaps := make([]float64, count+1)
	total := 0.0
	for index := range gaps {
		gaps[index] = rng.ExpFloat64()
		total += gaps[index]
	}

	timestamps := make([]time.Time, count)
	cumulative := 0.0
	for index := 0; index < count; index++ {
		cumulative += gaps[index]
		offset := time.Duration(float64(width) * cumulative / total)
		timestamps[index] = start.Add(offset)
	}
	sort.Slice(timestamps, func(i, j int) bool {
		return timestamps[i].Before(timestamps[j])
	})
	return timestamps
}
