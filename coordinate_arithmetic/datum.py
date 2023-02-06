import numpy as np
import torch as pt
import torch.utils.data as ptud


class CoordinateArithmeticSet(ptud.Dataset):
    operators = (
        lambda n1, n2, f: CoordinateArithmeticSet.coordinate_operation(f, n1, n2, 0),
        lambda n1, n2, f: CoordinateArithmeticSet.coordinate_operation(f, n1, n2, 1),
        lambda n1, n2, f: CoordinateArithmeticSet.coordinate_operation(f, n1, n2, 2),
        lambda n1, n2, f: CoordinateArithmeticSet.coordinate_operation(f, n1, n2, 3)
    )

    def __init__(self, n_slot, n_operation, n_example):
        super().__init__()
        self.inputs = []
        self.adjuncts = []

        for i in range(n_example):
            frames = [np.random.random([n_slot, 2])]
            operations = []

            for op in range(n_operation):
                nums = np.arange(n_slot)
                np.random.shuffle(nums)
                n1, n2 = nums

                oidx = np.random.choice(range(len(self.operators)))
                frame = self.operators[oidx](n1, n2, frames[-1])

                frames.append(frame)
                operations.append([n1, n2, oidx])

            self.inputs.append(np.array(frames, dtype='float32'))  # [(t,c,d)]
            self.adjuncts.append(np.array(operations, dtype='int32'))  # [(t,3)]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return pt.from_numpy(self.inputs[i]), pt.from_numpy(self.adjuncts[i])

    @staticmethod
    def coordinate_operation(state, number1, number2, operation):
        state2 = np.copy(state)
        if operation == 0:
            state2[number1, 0] += state2[number2, 0]
        elif operation == 1:
            state2[number1, 1] += state2[number2, 1]
        elif operation == 2:
            state2[number1, 0] -= state2[number2, 0]
        elif operation == 3:
            state2[number1, 1] -= state2[number2, 1]
        else:
            raise NotImplementedError
        return state2
