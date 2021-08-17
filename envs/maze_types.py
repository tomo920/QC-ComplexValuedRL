import numpy as np

def get_maze(type):
    if type == 1:
        legal_states = [np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.array([3.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([3.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([4.0, 1.0]), np.array([5.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([2.0, 2.0]), np.array([5.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([3.0, 4.0]), np.array([4.0, 4.0]), np.array([5.0, 4.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 6
        observation_list[legal_states[1]] = 7
        observation_list[legal_states[2]] = 3
        observation_list[legal_states[3]] = 3
        observation_list[legal_states[4]] = 4
        observation_list[legal_states[5]] = 5
        observation_list[legal_states[6]] = 1
        observation_list[legal_states[7]] = 2
        observation_list[legal_states[8]] = 3

        o_num = 7

        start_state = np.array([1.0, 2.0])
        goal_state = np.array([4.0, 2.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 2, 3]
        legal_action_list[4] = [0, 2, 3]
        legal_action_list[5] = [1]
        legal_action_list[6] = [0, 2]
        legal_action_list[7] = [0, 1]
    elif type == 2:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([3.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([3.0, 3.0]), np.array([4.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([3.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([4.0, 1.0]), np.array([5.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([4.0, 2.0]), np.array([5.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([5.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([4.0, 4.0]), np.array([5.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([4.0, 5.0]), np.array([5.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 10
        observation_list[legal_states[1]] = 11
        observation_list[legal_states[2]] = 8
        observation_list[legal_states[3]] = 9
        observation_list[legal_states[4]] = 5
        observation_list[legal_states[5]] = 5
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 3
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 1
        observation_list[legal_states[12]] = 2

        o_num = 11

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([4.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [1, 3]
        legal_action_list[3] = [0, 3]
        legal_action_list[4] = [0, 1, 2]
        legal_action_list[5] = [1, 2, 3]
        legal_action_list[6] = [0, 2, 3]
        legal_action_list[7] = [1]
        legal_action_list[8] = [0, 2]
        legal_action_list[9] = [0, 1, 3]
        legal_action_list[10] = [0, 2]
        legal_action_list[11] = [1, 2]
    elif type == 3:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]), np.array([5.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([3.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0]), np.array([5.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 9
        observation_list[legal_states[1]] = 2
        observation_list[legal_states[2]] = 11
        observation_list[legal_states[3]] = 4
        observation_list[legal_states[4]] = 9
        observation_list[legal_states[5]] = 10
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 6
        observation_list[legal_states[8]] = 7
        observation_list[legal_states[9]] = 8
        observation_list[legal_states[10]] = 4
        observation_list[legal_states[11]] = 1
        observation_list[legal_states[12]] = 5
        observation_list[legal_states[13]] = 6
        observation_list[legal_states[14]] = 1
        observation_list[legal_states[15]] = 2
        observation_list[legal_states[16]] = 3
        observation_list[legal_states[17]] = 4

        o_num = 11

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [0, 1, 3]
        legal_action_list[4] = [1]
        legal_action_list[5] = [1, 2]
        legal_action_list[6] = [2, 3]
        legal_action_list[7] = [0]
        legal_action_list[8] = [0, 1, 2, 3]
        legal_action_list[9] = [0, 2]
        legal_action_list[10] = [1, 3]
        legal_action_list[11] = [0, 1, 2]
    elif type == 4:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([3.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([3.0, 3.0]), np.array([4.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([3.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([4.0, 1.0]), np.array([5.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([4.0, 2.0]), np.array([5.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([5.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([4.0, 4.0]), np.array([5.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([4.0, 5.0]), np.array([5.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 7
        observation_list[legal_states[1]] = 9
        observation_list[legal_states[2]] = 7
        observation_list[legal_states[3]] = 8
        observation_list[legal_states[4]] = 4
        observation_list[legal_states[5]] = 4
        observation_list[legal_states[6]] = 5
        observation_list[legal_states[7]] = 6
        observation_list[legal_states[8]] = 1
        observation_list[legal_states[9]] = 3
        observation_list[legal_states[10]] = 4
        observation_list[legal_states[11]] = 1
        observation_list[legal_states[12]] = 2

        o_num = 9

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([4.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [1, 3]
        legal_action_list[3] = [0, 1, 2]
        legal_action_list[4] = [1, 2, 3]
        legal_action_list[5] = [0, 2, 3]
        legal_action_list[6] = [1]
        legal_action_list[7] = [0, 2]
        legal_action_list[8] = [0, 1, 3]
        legal_action_list[9] = [1, 2]
    elif type == 5:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 11
        observation_list[legal_states[1]] = 12
        observation_list[legal_states[2]] = 13
        observation_list[legal_states[3]] = 9
        observation_list[legal_states[4]] = 10
        observation_list[legal_states[5]] = 6
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 8
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 6
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 13

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 3]
        legal_action_list[4] = [0, 3]
        legal_action_list[5] = [1, 2]
        legal_action_list[6] = [2, 3]
        legal_action_list[7] = [0, 2, 3]
        legal_action_list[8] = [1]
        legal_action_list[9] = [0, 2]
        legal_action_list[10] = [1, 3]
        legal_action_list[11] = [0, 2]
        legal_action_list[12] = [0, 1]
        legal_action_list[13] = [1, 2]
    elif type == 6:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 8
        observation_list[legal_states[1]] = 10
        observation_list[legal_states[2]] = 11
        observation_list[legal_states[3]] = 8
        observation_list[legal_states[4]] = 9
        observation_list[legal_states[5]] = 5
        observation_list[legal_states[6]] = 5
        observation_list[legal_states[7]] = 6
        observation_list[legal_states[8]] = 7
        observation_list[legal_states[9]] = 1
        observation_list[legal_states[10]] = 4
        observation_list[legal_states[11]] = 5
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 11

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 3]
        legal_action_list[4] = [1, 2]
        legal_action_list[5] = [2, 3]
        legal_action_list[6] = [0, 2, 3]
        legal_action_list[7] = [1]
        legal_action_list[8] = [0, 2]
        legal_action_list[9] = [1, 3]
        legal_action_list[10] = [0, 1]
        legal_action_list[11] = [1, 2]
    elif type == 7:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([3.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([3.0, 3.0]), np.array([4.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([3.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([4.0, 1.0]), np.array([5.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([4.0, 2.0]), np.array([5.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([5.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([4.0, 4.0]), np.array([5.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([4.0, 5.0]), np.array([5.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 1
        observation_list[legal_states[1]] = 10
        observation_list[legal_states[2]] = 8
        observation_list[legal_states[3]] = 9
        observation_list[legal_states[4]] = 5
        observation_list[legal_states[5]] = 5
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 3
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 1
        observation_list[legal_states[12]] = 2

        o_num = 10

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([4.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 2, 3]
        legal_action_list[2] = [1, 3]
        legal_action_list[3] = [0, 3]
        legal_action_list[4] = [0, 1, 2]
        legal_action_list[5] = [1, 2, 3]
        legal_action_list[6] = [0, 2, 3]
        legal_action_list[7] = [1]
        legal_action_list[8] = [0, 2]
        legal_action_list[9] = [0, 1, 3]
        legal_action_list[10] = [1, 2]
    elif type == 8:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 11
        observation_list[legal_states[1]] = 2
        observation_list[legal_states[2]] = 12
        observation_list[legal_states[3]] = 9
        observation_list[legal_states[4]] = 10
        observation_list[legal_states[5]] = 6
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 8
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 6
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 12

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 3]
        legal_action_list[4] = [0, 3]
        legal_action_list[5] = [1, 2]
        legal_action_list[6] = [2, 3]
        legal_action_list[7] = [0, 2, 3]
        legal_action_list[8] = [1]
        legal_action_list[9] = [0, 2]
        legal_action_list[10] = [1, 3]
        legal_action_list[11] = [0, 2]
        legal_action_list[12] = [1, 2]
    elif type == 9:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 1
        observation_list[legal_states[1]] = 11
        observation_list[legal_states[2]] = 12
        observation_list[legal_states[3]] = 9
        observation_list[legal_states[4]] = 10
        observation_list[legal_states[5]] = 6
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 8
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 6
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 12

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 2, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 3]
        legal_action_list[4] = [0, 3]
        legal_action_list[5] = [1, 2]
        legal_action_list[6] = [2, 3]
        legal_action_list[7] = [0, 2, 3]
        legal_action_list[8] = [1]
        legal_action_list[9] = [0, 2]
        legal_action_list[10] = [1, 3]
        legal_action_list[11] = [0, 1]
        legal_action_list[12] = [1, 2]
    elif type == 10:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 10
        observation_list[legal_states[1]] = 11
        observation_list[legal_states[2]] = 12
        observation_list[legal_states[3]] = 4
        observation_list[legal_states[4]] = 9
        observation_list[legal_states[5]] = 6
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 8
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 6
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 12

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 3]
        legal_action_list[4] = [0, 2, 3]
        legal_action_list[5] = [1, 2]
        legal_action_list[6] = [2, 3]
        legal_action_list[7] = [0, 2, 3]
        legal_action_list[8] = [1]
        legal_action_list[9] = [0, 2]
        legal_action_list[10] = [1, 3]
        legal_action_list[11] = [0, 2]
        legal_action_list[12] = [1, 2]
    elif type == 11:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 11
        observation_list[legal_states[1]] = 12
        observation_list[legal_states[2]] = 3
        observation_list[legal_states[3]] = 9
        observation_list[legal_states[4]] = 10
        observation_list[legal_states[5]] = 6
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 8
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 6
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 12

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 2, 3]
        legal_action_list[4] = [0, 3]
        legal_action_list[5] = [1, 2]
        legal_action_list[6] = [2, 3]
        legal_action_list[7] = [0, 2, 3]
        legal_action_list[8] = [1]
        legal_action_list[9] = [0, 2]
        legal_action_list[10] = [1, 3]
        legal_action_list[11] = [0, 2]
        legal_action_list[12] = [1, 2]
    elif type == 12:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 10
        observation_list[legal_states[1]] = 11
        observation_list[legal_states[2]] = 12
        observation_list[legal_states[3]] = 9
        observation_list[legal_states[4]] = 5
        observation_list[legal_states[5]] = 6
        observation_list[legal_states[6]] = 6
        observation_list[legal_states[7]] = 7
        observation_list[legal_states[8]] = 8
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 5
        observation_list[legal_states[11]] = 6
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 12

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1]
        legal_action_list[3] = [1, 3]
        legal_action_list[4] = [0, 3]
        legal_action_list[5] = [1, 2, 3]
        legal_action_list[6] = [2, 3]
        legal_action_list[7] = [0, 2, 3]
        legal_action_list[8] = [1]
        legal_action_list[9] = [0, 2]
        legal_action_list[10] = [0, 2]
        legal_action_list[11] = [0, 1]
        legal_action_list[12] = [1, 2]
    elif type == 13:
        legal_states = [np.array([2.0, 1.0]), np.array([3.0, 1.0]), np.array([4.0, 1.0]),
                        np.array([1.0, 2.0]), np.array([2.0, 2.0]), np.array([4.0, 2.0]),
                        np.array([1.0, 3.0]), np.array([4.0, 3.0]), np.array([5.0, 3.0]),
                        np.array([1.0, 4.0]), np.array([2.0, 4.0]), np.array([4.0, 4.0]),
                        np.array([2.0, 5.0]), np.array([3.0, 5.0]), np.array([4.0, 5.0])]
        illegal_states = [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0]), np.array([4.0, 0.0]), np.array([5.0, 0.0]), np.array([6.0, 0.0]),
                          np.array([0.0, 1.0]), np.array([1.0, 1.0]), np.array([5.0, 1.0]), np.array([6.0, 1.0]),
                          np.array([0.0, 2.0]), np.array([3.0, 2.0]), np.array([5.0, 2.0]), np.array([6.0, 2.0]),
                          np.array([0.0, 3.0]), np.array([2.0, 3.0]), np.array([3.0, 3.0]), np.array([6.0, 3.0]),
                          np.array([0.0, 4.0]), np.array([3.0, 4.0]), np.array([5.0, 4.0]), np.array([6.0, 4.0]),
                          np.array([0.0, 5.0]), np.array([1.0, 5.0]), np.array([5.0, 5.0]), np.array([6.0, 5.0]),
                          np.array([0.0, 6.0]), np.array([1.0, 6.0]), np.array([2.0, 6.0]), np.array([3.0, 6.0]), np.array([4.0, 6.0]), np.array([5.0, 6.0]), np.array([6.0, 6.0])]
        legal_states = [state.tostring() for state in legal_states]
        illegal_states = [state.tostring() for state in illegal_states]

        observation_list = {}
        observation_list[legal_states[0]] = 10
        observation_list[legal_states[1]] = 9
        observation_list[legal_states[2]] = 11
        observation_list[legal_states[3]] = 8
        observation_list[legal_states[4]] = 9
        observation_list[legal_states[5]] = 5
        observation_list[legal_states[6]] = 5
        observation_list[legal_states[7]] = 6
        observation_list[legal_states[8]] = 7
        observation_list[legal_states[9]] = 4
        observation_list[legal_states[10]] = 2
        observation_list[legal_states[11]] = 5
        observation_list[legal_states[12]] = 1
        observation_list[legal_states[13]] = 2
        observation_list[legal_states[14]] = 3

        o_num = 11

        start_state = np.array([1.0, 3.0])
        goal_state = np.array([5.0, 3.0])

        legal_action_list = {}
        legal_action_list[1] = [0, 3]
        legal_action_list[2] = [0, 1, 2]
        legal_action_list[3] = [1, 3]
        legal_action_list[4] = [0, 3]
        legal_action_list[5] = [2, 3]
        legal_action_list[6] = [0, 2, 3]
        legal_action_list[7] = [1]
        legal_action_list[8] = [0, 2]
        legal_action_list[9] = [1, 3]
        legal_action_list[10] = [0, 2]
        legal_action_list[11] = [1, 2]

    return legal_states, illegal_states, observation_list, o_num, start_state, goal_state, legal_action_list
