package rbm

import "time"
import "fmt"

type StepMonitor struct {
	start time.Time
	last float64
	i, n int
}

func (pm *StepMonitor) Tick(i, n int) bool {

	pm.i = i
	pm.n = n

	if pm.start.IsZero() {
		pm.start = time.Now()
		return false
	}
	
	since := time.Since(pm.start).Seconds()

	if since > pm.last + 1.5 {
		pm.last = since
		return true
	}
	
	return false
}

func (pm *StepMonitor) String() string {
	
	//for j := pm.n; j > 0; j >>= 1 { sz++}
	since := time.Since(pm.start).Seconds()

	percent := 100 * pm.i / pm.n
	remaining := float64(pm.n - pm.i) * since / float64(pm.i)

	return fmt.Sprintf(
		"%02d%% complete, %s remaining, #%d",
		percent, time.Duration(remaining) * time.Second, pm.i,
	)
}