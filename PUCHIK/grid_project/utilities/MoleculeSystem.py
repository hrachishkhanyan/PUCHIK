import abc


class MoleculeSystem(abc.ABC):
    """ Every PUCHIK class should derive from this in the future. """
    @abc.abstractmethod
    def __init__(self, traj: str, top=None):
        ...

    @abc.abstractmethod
    def select_atoms(self, selection: str) -> None:
        ...
