import numpy as np
from sklearn.neighbors import KDTree
from rtree import index
from random import choice
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:
    pass


class Particle:
    def __init__(self, radius, position):
        self.radius = radius
        self.position = position
        self.colliding = False
        self.locked = False
        self.volume = sphere_volume(radius)
        self.reset_displacement()

    def displace_by(self, other):
        A = self.position - other.position
        d = np.sqrt(np.sum(A ** 2))
        collision_radius = self.radius + other.radius
        f = Particle.get_displacement_force(collision_radius, d)
        f_inert = f * other.volume / (other.volume + self.volume)
        A_norm = A / d
        self.displacement = self.displacement + A_norm * f_inert * collision_radius
        # print()
        # print(self.id, "being displaced by", other.id)
        # print("---")
        # print("Initial: Displacing particles were {} cr away from eachother".format(d / collision_radius))
        # print("Repulsion: Particles will now move {} cr away from each other".format(f))
        # print("Intertia: I make up {} of the total volume".format(self.volume / (other.volume + self.volume)))
        # print("Intertia: This particle moving {} cr".format(f_inert))
        # print("Final: ".format(d / collision_radius + f))
        # print("Displacement: particles end up {} cr away from eachother", A_norm * f_inert * collision_radius)

    def displace(self):
        # TODO: STAY INSIDE OF PARTNER RADIUS
        self.position = self.position + self.displacement
        self.reset_displacement()

    def reset_displacement(self):
        self.displacement = np.zeros((len(self.position)))

    @staticmethod
    def get_displacement_force(radius, distance):
        if distance == 0.:
            return 0.9
        return min(0.9, 0.3 / ((distance / radius) ** 2))


class Neighbourhood:
    def __init__(self, epicenter, neighbours, neighbour_radius, partners, partner_radius):
        self.epicenter = epicenter
        self.neighbours = neighbours
        self.neighbour_radius = neighbour_radius
        self.partners = partners
        self.partner_radius = partner_radius

    def get_overlap(self):
        overlap = 0
        neighbours = self.neighbours
        n_neighbours = len(neighbours)
        for partner in self.partners:
            for neighbour in self.neighbours:
                if partner.id == neighbour.id:
                    continue
                overlap -= min(0, distance(partner.position, neighbour.position) - partner.radius - neighbour.radius)
        return overlap

    def colliding(self):
        neighbours = self.neighbours
        for partner in self.partners:
            for neighbour in self.neighbours:
                if partner.id == neighbour.id:
                    continue
                if distance(partner.position, neighbour.position) - partner.radius - neighbour.radius < 0:
                    return True
        return False


class ParticleVoxel:
    def __init__(self, origin, dimensions):
        self.origin = np.array(origin)
        self.size = np.array(dimensions)


class ParticleSystem():
    def __init__(self):
        self.particle_types = []
        self.voxels = []

    def fill(self, volume, voxels, particles):
        # Amount of spatial dimensions
        self.dimensions = len(volume)
        #
        self.size = volume
        # Extend list of particle types in the system
        self.particle_types.extend(particles)
        # Max particle type radius
        self.max_radius = max([pt["radius"] for pt in self.particle_types])
        self.min_radius = min([pt["radius"] for pt in self.particle_types])
        # Set initial radius for collision/rearrangement to 2 times the largest particle type radius
        self.search_radius = self.max_radius * 2
        # Create a list of voxels where the particles can be placed.
        self.voxels.extend([ParticleVoxel(v[0], v[1]) for v in voxels])
        # Reset particles
        self.particles = []
        for particle_type in self.particle_types:
            radius = particle_type["radius"]
            placement_voxels = particle_type["voxels"]
            particle_count = particle_type["count"]
            # Generate a matrix with random positions for the particles
            # Add an extra dimension to determine in which voxels to place the particles
            placement_matrix = np.random.rand(particle_count, self.dimensions + 1)
            # Generate each particle
            for positions in placement_matrix:
                # Determine the voxel to be placed in.
                particle_voxel_id = int(positions[0] * len(placement_voxels))
                particle_voxel = self.voxels[particle_voxel_id]
                # Translate the particle into the voxel based on the remaining dimensions
                particle_position = particle_voxel.origin + positions[1:] * particle_voxel.size
                # Store the particle object
                self.add_particle(radius, particle_position)

    def freeze(self):
        self.positions = np.array([p.position for p in self.particles])
        self.radii = [p.radius for p in self.particles]
        self.tree = KDTree(self.positions)

    def find_colliding_particles(self):
        if not hasattr(self, "tree"):
            self.freeze()
        # Do an O(n * log(n)) search of all particles by the maximum radius
        neighbours = self.tree.query_radius(self.positions, r=self.search_radius, return_distance=True)
        neighbour_ids = neighbours[0]
        neighbour_distances = neighbours[1]
        for i in range(len(neighbour_ids)):
            distances = neighbour_distances[i]
            if len(distances) == 1:  # No neighbours
                continue
            ids = neighbour_ids[i]
            particle = self.particles[i]
            for j in range(len(ids)):
                neighbour_id = ids[j]
                if neighbour_id == i:
                    continue
                neighbour = self.particles[neighbour_id]
                min_radius = particle.radius + neighbour.radius
                if distances[j] <= min_radius:
                    particle.colliding = True
                    neighbour.colliding = True
        self.colliding_particles = list(filter(lambda p: p.colliding, self.particles))
        self.colliding_count = len(self.colliding_particles)
        return self.colliding_particles

    def solve_collisions(self):
        self.find_colliding_particles()
        while self.colliding_count > 0:
            print("Untangling {} collisions".format(self.colliding_count))
            i = 0
            t = self.colliding_count
            for epicenter_particle in self.colliding_particles:
                i += 1
                neighbourhood = self.find_neighbourhood(epicenter_particle)
                self.resolve_neighbourhood(neighbourhood)
                print(i, "/", t, end="\r")
            # Double check that there's no collisions left
            self.freeze()
            self.find_colliding_particles()
            # print("Neighbourhood solved.", len(self.colliding_particles), "colliding particles remaining.")
        # print("Tadaa! Placement complete.")

    def resolve_neighbourhood(self, neighbourhood):
        # for neighbour in neighbourhood.neighbours:
        #     neighbour.locked = True
        # for partner in neighbourhood.partners:
        #     partner.locked = False
        i = 0
        # print("Solving neighbourhood", neighbourhood.epicenter.id)
        # print("---")
        stuck = False
        overlap = 0.
        while neighbourhood.colliding():
            i += 1
            overlap = neighbourhood.get_overlap()
            # print(i, "Neighbourhood overlap:", overlap)
            for partner in neighbourhood.partners:
                for neighbour in neighbourhood.neighbours:
                    if partner.id == neighbour.id:
                        continue
                    partner.displace_by(neighbour)
            for partner in neighbourhood.partners:
                partner.displace()
            overlap = neighbourhood.get_overlap()
            # print()
            if i > 100:
                stuck = True
                print("STUCK")
                break
        if not stuck:
            self.colliding_count -= len(neighbourhood.partners)
            for partner in neighbourhood.partners:
                partner.colliding = False

    def find_neighbourhood(self, particle):
        epicenter = particle.position
        # print("Finding collision neighbourhood for particle", particle.id)
        neighbourhood_radius = self.max_radius * 2
        neighbourhood_ok = False
        expansions = 0
        while not neighbourhood_ok:
            expansions += 1
            neighbourhood_radius += self.max_radius / min(expansions, 6)
            neighbour_ids = self.tree.query_radius([epicenter], r=neighbourhood_radius)[0]
            neighbours = [self.particles[id] for id in neighbour_ids]
            neighbourhood_packing_factor = self.get_packing_factor(neighbours, sphere_volume(neighbourhood_radius))
            partner_radius = neighbourhood_radius - self.max_radius
            partner_ids = self.tree.query_radius([epicenter], r=partner_radius)[0]
            partners = [self.particles[id] for id in partner_ids]
            partner_packing_factor = self.get_packing_factor(partners, sphere_volume(partner_radius))
            partners = list(filter(lambda p: not p.locked and p.colliding, partners))
            neighbourhood_ok = neighbourhood_packing_factor < 0.5 and partner_packing_factor < 0.5
            if expansions > 100:
                print("ERROR! Unable to find suited neighbourhood around", epicenter)
                exit()
        # print("Neighbourhood of {} particles with radius {} and packing factor of {}. Found after {} expansions.".format(
        #     len(neighbour_ids), neighbourhood_radius, partner_packing_factor, expansions
        # ))
        # print(len(partner_ids), "particles will be moved.")
        return Neighbourhood(epicenter, neighbours, neighbourhood_radius, partners, partner_radius)

    def add_particle(self, radius, position):
        particle = Particle(radius, position)
        particle.id = len(self.particles)
        self.particles.append(particle)

    def add_particles(self, radius, positions):
        for position in positions:
            self.add_particle(radius, position)

    def remove_particles(self, particles_id):
        # Remove particles with a certain id
        for index in sorted(particles_id,  reverse = True):
            del self.particles[index]

    def deintersect(self, nearest_neighbours=None):
        if nearest_neighbours is None:
            nearest_neighbours = self.estimate_nearest_neighbours()

    def get_packing_factor(self, particles=None, volume=None):
        if particles is None:
            particles_volume = np.sum([p["count"] * sphere_volume(p["radius"]) for p in self.particle_types])
        else:
            particles_volume = np.sum([sphere_volume(p.radius) for p in particles])
        total_volume = np.product(self.size) if volume is None else volume
        return particles_volume / total_volume

    def prune(self,colliding_particles):
        # Among the colliding_particles, check for the ones that have been reallocated outside the volume
        # and remove from particle system. Rtree algorithm is used to check for particles outside  the volume,
        # taking the voxels as the rtree elements and the particle point as the volume for which to check the intersection
        property = index.Property()
        property.dimension = 3
        idx = index.Index(properties=property, interleaved=True)
        for x in range(len(self.voxels)):
            print(self.voxels[x].origin[0])
            idx.insert(x, (self.voxels[x].origin[0], self.voxels[x].origin[1], self.voxels[x].origin[2], self.voxels[x].origin[0]+self.voxels[x].size[0], self.voxels[x].origin[1]+self.voxels[x].size[1], self.voxels[x].origin[2]+self.voxels[x].size[2]))
        colliding_external = list(filter(lambda p: not(list(idx.intersection((p.position[0],p.position[1],p.position[2],p.position[0],p.position[1],p.position[2])))), colliding_particles))
        colliding_particles_id = list(map(lambda c: c.id, colliding_external))
        pruned = self.remove_particles(colliding_particles_id)
        number_pruned = len(colliding_particles_id)
        return number_pruned


class LargeParticleSystem(ParticleSystem):
    def __init__(self):
        ParticleSystem.__init__()



    def placing(self):
        pass


    def fill(self):
        super().fill()


    def solve_collisions(self):
        #todo: take smaller particle systems

        super().solve_collisions()





def plot_particle_system(system):
    nc_particles = list(filter(lambda p: not p.colliding, system.particles))
    c_particles = list(filter(lambda p: p.colliding, system.particles))
    nc_trace = get_particles_trace(nc_particles)
    c_trace = get_particles_trace(c_particles, marker=dict(color="rgba(200, 100, 0, 1)", size=2))
    fig = go.Figure(data=[c_trace, nc_trace])
    if system.dimensions == 3:
        fig.update_layout(scene_aspectmode='cube')
        fig.layout.scene.xaxis.range = [0., system.size[0]]
        fig.layout.scene.yaxis.range = [0., system.size[1]]
        fig.layout.scene.zaxis.range = [0., system.size[2]]
    fig.show()


def get_particles_trace(particles, dimensions=3, axes={'x': 0, 'y': 1, 'z': 2}, **kwargs):
    trace_kwargs = {"mode": "markers", "marker": {"color": "rgba(100, 100, 100, 0.7)", "size": 1}}
    trace_kwargs.update(kwargs)
    if dimensions > 3:
        raise Exception("Maximum 3 dimensional plots. Unless you have mutant eyes.")
    elif dimensions == 3:
        return go.Scatter3d(
            x=list(map(lambda p: p.position[axes['x']], particles)),
            y=list(map(lambda p: p.position[axes['y']], particles)),
            z=list(map(lambda p: p.position[axes['z']], particles)),
            **trace_kwargs
        )
    elif dimensions == 2:
        return go.Scatter(
            x=list(map(lambda p: p.position[axes['x']], particles)),
            y=list(map(lambda p: p.position[axes['y']], particles)),
            **trace_kwargs
        )
    elif dimensions == 1:
        return go.Scatter(
            x=list(map(lambda p: p.position[axes['x']], particles)),
            **trace_kwargs
        )


def plot_detailed_system(system):
    fig = go.Figure()
    fig.update_layout(showlegend=False)
    for particle in system.particles:
        trace = get_particle_trace(particle)
        fig.add_trace(trace)
    fig.update_layout(scene_aspectmode='data')
    fig.update_layout(scene = dict(
        xaxis = dict(tick0=0, dtick=system.voxels[0].size[0],),         # Use the size of the first voxel to set ticks of axes
                     yaxis = dict(tick0=650, dtick=system.voxels[0].size[1],),
                     zaxis = dict(tick0=800, dtick=system.voxels[0].size[2],)
                     ))
    fig.show()
    return fig


def get_particle_trace(particle):
    theta = np.linspace(0, 2*np.pi, 10)
    phi = np.linspace(0, np.pi, 10)
    x = np.outer(np.cos(theta), np.sin(phi)) * particle.radius + particle.position[0]
    y = np.outer(np.sin(theta), np.sin(phi)) * particle.radius + particle.position[1]
    z = np.outer(np.ones(10), np.cos(phi)) * particle.radius + particle.position[2]
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros(10) + int(particle.colliding),
        colorscale=[[0, "rgb(100, 100, 100)"], [1, "rgb(200, 100, 0)"]],
        opacity=0.5 + 0.5 * int(particle.colliding),
        showscale=False
    )


def sphere_volume(radius):
    return 4 / 3 * np.pi * radius ** 3


def distance(a, b):
    return np.sqrt(np.sum((b - a) ** 2))


class AdaptiveNeighbourhood(ParticleSystem):
    def find_neighbourhood(self, particle):
        epicenter = particle.position
        # print("Finding collision neighbourhood for particle", particle.id)
        precautious_radius = particle.radius + self.max_radius
        partner_ids = self.tree.query_radius([epicenter], r=precautious_radius)[0]
        if len(partner_ids) == 0:
            return Neighbourhood(epicenter, [], precautious_radius, [], precautious_radius)
        # partner_radius = rA + rB where B is the largest particle around.
        partner_radius = np.max([self.particles[id].radius if id != particle.id else 0. for id in partner_ids])
        neighbourhood_ok = False
        expansions = 0
        while not neighbourhood_ok:
            expansions += 1
            partner_radius += particle.radius / min(expansions, 6)
            partner_ids = self.tree.query_radius([epicenter], r=partner_radius)[0]
            partners = [self.particles[id] for id in partner_ids]
            partner_packing_factor = self.get_packing_factor(partners, sphere_volume(partner_radius))
            if partner_packing_factor > 0.5:
                continue

            neighbourhood_radius = partner_radius + self.max_radius
            neighbour_ids = self.tree.query_radius([epicenter], r=neighbourhood_radius)[0]
            neighbours = [self.particles[id] for id in neighbour_ids]
            strictly_neighbours = list(filter(lambda n: n.id not in partner_ids, neighbours))
            if len(strictly_neighbours) > 0:
                max_neighbour_radius = np.max([n.radius for n in strictly_neighbours])
                if max_neighbour_radius != self.max_radius:
                    neighbourhood_radius = partner_radius + max_neighbour_radius
                    neighbour_ids = self.tree.query_radius([epicenter], r=neighbourhood_radius)[0]
                    neighbours = [self.particles[id] for id in neighbour_ids]
            neighbourhood_packing_factor = self.get_packing_factor(neighbours, sphere_volume(neighbourhood_radius))
            neighbourhood_ok = neighbourhood_packing_factor < 0.5
            if expansions > 100:
                print("ERROR! Unable to find suited neighbourhood around", epicenter)
                exit()
        # print("Neighbourhood of {} particles with radius {} and packing factor of {}. Found after {} expansions.".format(
        #     len(neighbour_ids), neighbourhood_radius, partner_packing_factor, expansions
        # ))
        # print(len(partner_ids), "particles will be moved.")
        return Neighbourhood(epicenter, neighbours, neighbourhood_radius, partners, partner_radius)


class SmallestNeighbourhood(ParticleSystem):

    def find_neighbourhood(self, particle):
        epicenter = particle.position
        # print("Finding collision neighbourhood for particle", particle.id)
        neighbourhood_radius = particle.radius + self.min_radius
        neighbourhood_ok = False
        expansions = 0
        while not neighbourhood_ok:
            expansions += 1
            neighbourhood_radius += self.min_radius
            neighbour_ids = self.tree.query_radius([epicenter], r=neighbourhood_radius)[0]
            if len(neighbour_ids) == 0:
                return Neighbourhood(epicenter, [], 0, [], 0)
            neighbours = [self.particles[id] for id in neighbour_ids]
            max_neighbour_radius = np.max([n.radius for n in neighbours])
            neighbourhood_packing_factor = self.get_packing_factor(neighbours, sphere_volume(neighbourhood_radius))
            partner_radius = neighbourhood_radius - max_neighbour_radius
            partner_ids = self.tree.query_radius([epicenter], r=partner_radius)[0]
            partners = [self.particles[id] for id in partner_ids]
            partner_packing_factor = self.get_packing_factor(partners, sphere_volume(partner_radius))
            partners = list(filter(lambda p: not p.locked and p.colliding, partners))
            neighbourhood_ok = neighbourhood_packing_factor < 0.5 and partner_packing_factor < 0.5
            if expansions > 100:
                print("ERROR! Unable to find suited neighbourhood around", epicenter)
                exit()
        # print("Neighbourhood of {} particles with radius {} and packing factor of {}. Found after {} expansions.".format(
        #     len(neighbour_ids), neighbourhood_radius, partner_packing_factor, expansions
        # ))
        # print(len(partner_ids), "particles will be moved.")
        return Neighbourhood(epicenter, neighbours, neighbourhood_radius, partners, partner_radius)
