import shortuuid
import itertools
from bsb import config
from bsb.exceptions import DatasetNotFoundError
from bsb.config import types
from bsb.simulation.cell import CellModel


@config.node
class NestCell(CellModel):
    neuron_model = config.attr(type=str)
    constants = config.dict(type=types.any_())

    def create_population(self, simdata):
        import nest

        population = nest.Create(self.neuron_model, len(self.get_placement_set()))
        self.set_constants(population)
        self.set_parameters(population)
        return population

    def set_constants(self, population):
        population.set(self.constants)

    def set_parameters(self, population):
        ps = self.get_placement_set()
        for param in self.parameters:
            population.set(param.name, param.get_value(ps))
