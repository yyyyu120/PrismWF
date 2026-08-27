package detorrent

import (
	"fmt"
	"strconv"

	pt "git.torproject.org/pluggable-transports/goptlib.git"
	"github.com/websitefingerprinting/wfdef.git/transports/defconn"
)

type jsonServerState struct {
	defconn.JsonServerState
	Budget        int    `json:"budget"`
	GeneratorAddr string `json:"generator-addr"`
}

type serverState struct {
	defconn.DefConnServerState
	budget        int
	generatorAddr string
}

func (state *serverState) clientString() string {
	return state.DefConnServerState.ClientString() +
		fmt.Sprintf("%s=%d %s=%s", budgetArg, state.budget, generatorAddrArg, state.generatorAddr)
}

func serverStateFromArgs(stateDir string, args *pt.Args) (*serverState, error) {
	baseState, err := defconn.ServerStateFromArgsInternal(stateDir, defconn.StateFile, args)
	if err != nil {
		return nil, err
	}

	budgetString, ok := args.Get(budgetArg)
	if !ok {
		return nil, fmt.Errorf("missing argument '%s'", budgetArg)
	}
	budget, err := strconv.Atoi(budgetString)
	if err != nil || budget <= 0 {
		return nil, fmt.Errorf("invalid %s '%s'", budgetArg, budgetString)
	}

	generatorAddr, ok := args.Get(generatorAddrArg)
	if !ok || generatorAddr == "" {
		generatorAddr = defaultGeneratorAddr
	}

	jsonState := jsonServerState{
		JsonServerState: baseState,
		Budget:          budget,
		GeneratorAddr:   generatorAddr,
	}
	return serverStateFromJSON(stateDir, &jsonState)
}

func serverStateFromJSON(stateDir string, jsonState *jsonServerState) (*serverState, error) {
	baseState, err := defconn.ServerStateFromJsonServerStateInternal(jsonState)
	if err != nil {
		return nil, err
	}
	if jsonState.Budget <= 0 {
		return nil, fmt.Errorf("invalid %s '%d'", budgetArg, jsonState.Budget)
	}
	if jsonState.GeneratorAddr == "" {
		jsonState.GeneratorAddr = defaultGeneratorAddr
	}

	state := &serverState{
		DefConnServerState: baseState,
		budget:             jsonState.Budget,
		generatorAddr:      jsonState.GeneratorAddr,
	}
	if err := defconn.NewBridgeFile(stateDir, defconn.BridgeFile, state.clientString()); err != nil {
		return nil, err
	}
	return state, defconn.WriteJSONServerState(stateDir, defconn.StateFile, jsonState)
}
