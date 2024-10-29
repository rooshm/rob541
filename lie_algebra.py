import numpy as np
import numdifftools as nd
from repgroup import RepGroup, RepGroupElement
from vector import VectorBases, TangentVector

def derivative_group_action_bases(direction: RepGroupElement, config: RepGroupElement, action = "left"):
    def func(direction, config, i, action, delta):
        if isinstance(delta, np.ndarray):
            delta = delta.flatten()[0]
        flattened_direction = direction.derepresentation.flatten()
        flattened_direction[i] = flattened_direction[i] + delta
        new_basis = RepGroupElement(value=flattened_direction, group=direction.group)
        if action == "left":
            return new_basis.left_action(config).value
        elif action == "right":
            return new_basis.right_action(config).value
        else:
            raise ValueError("Invalid action")

    gtvs = []
    flattened_direction = direction.derepresentation.flatten()
    for i in range(flattened_direction.shape[0]):
        reduced_group_action = lambda delta: func(direction, config, i, action, delta)
        deriv = nd.Derivative(reduced_group_action)(0.)
        # print(f'deriv: {deriv} at config {config} for basis g{i}')
        gtv = GroupTangentVector(val=RepGroupElement(value=deriv, group=direction.group), config=config)
        gtvs.append(gtv)
    return gtvs

class GroupTangentVector(object):
    def __init__(self, val: RepGroupElement, config: RepGroupElement) -> None:
        self.config = config
        self.group = self.config.group
        self.value = val

    def flatten(self):
        return self.value.derepresentation.flatten().reshape(-1,1)

    def as_tv(self):
        return TangentVector(value=self.value.derepresentation, config=self.config.derepresentation)

    def __repr__(self) -> str:
        return f'GroupTangentVector(value: {self.value}, config: {self.config}, group: {self.group})'

    def left_velocity(self):
        bases = self.group.bases([self.group.identity_element]).evaluate(self.config, action="left").flatten()
        print(f'bases: {bases}')
        velocity = np.linalg.inv(bases) @ self.value
        return velocity.flatten()[0]

    def right_velocity(self):
        bases = self.group.bases([self.group.identity_element()]).evaluate(self.config, action="right").flatten()
        print(f'bases: {bases}')
        velocity = np.linalg.inv(bases) @ self.value
        return velocity.flatten()[0]

class GroupVectorBases(object):
    def __init__(self, vectors: list) -> None:
        self.vectors = vectors

    @property
    def elements(self):
        return self.vectors

    def __repr__(self) -> str:
        return f'GroupVectorBases(vectors: {self.vectors})'

    def evaluate(self, elem, action = "left", in_vb_form = True):
        """ Evaluate coordinate basis of elem at list of initialised configurations using action.
        in_vb_form: True returns list of TangentVectors for each basis useful for plotting
        else Returns nparray of GroupTangentVector of shape (configs, basis_vectors)"""
        group_tangent_bases = np.array([derivative_group_action_bases(direction=elem, config=vector, action=action) for vector in self.vectors], dtype=GroupTangentVector)
        if not in_vb_form:
            return group_tangent_bases
        # shape = (configs, basis_vectors)
        # per basis list
        vblist = []
        for i in range(0, group_tangent_bases.shape[1]):
            vb = VectorBases([gtv.as_tv() for gtv in group_tangent_bases[:, i]], inverse=group_tangent_bases[0][i].group.identity)
            vblist.append(vb)
        return vblist

class LieGroup(RepGroup):
    def __init__(self, represent, depresent, identity):
        super().__init__(represent, depresent, identity)
        self.bases = lambda h: GroupVectorBases([h])

    def element(self, value):
        return LieGroupElement(self, value, self.left_lifted_action, self.right_lifted_action)

    def identity_element(self):
        return self.element(self.identity)

    # Left lifted action: A function that takes in two instances of the “group element”
    # class and returns the matrix Th Lg . In Python, this function can be written
    # to automatically generate the lifted action matrix from the group action by
    # wrapping a lambda function around the operation g * h that locks in the
    # acting element g and exposes the current-configuration element h, then using
    # numdifftools.Jacobian to take the derivative with respect to the current-
    # configuration element.
    def left_lifted_action(self, g, h_config):
        group = g.group
        reprshape = group.identity_element().derepresentation.shape
        reduced_func = lambda h: g.left_action(group.element(h.reshape(reprshape))).derepresentation
        jacobian = nd.Jacobian(reduced_func)
        jacobian_matrix = jacobian(h_config.derepresentation)
        return jacobian_matrix

    # ii. Right lifted action: A function that takes in a second instance of the group
    # class and returns the matrix Tg Rh . In Python, this function can be written
    # to automatically generate the lifted action matrix from the group action by
    # wrapping a lambda function around the operation g * h that locks in the
    # acting element h and exposes the current-configuration element g, then using
    # numdifftools.Jacobian to take the derivative with respect to the current-
    # configuration element.
    def right_lifted_action(self, h, g_config):
        group = h.group
        reprshape = group.identity_element().derepresentation.shape
        reduced_func = lambda g: h.right_action(group.element(g.reshape(reprshape))).derepresentation
        jacobian = nd.Jacobian(reduced_func)
        jacobian_matrix = jacobian(g_config.derepresentation)
        return jacobian_matrix

    # def TL(self, gdotatq):
    #     """maps gdotatg to hgdotatg"""
    #     g = gdotatq.config
    #     gval = g.value
    #     func = lambda gval: self.group.operation(self.value, gval)
    #     TgLh = nd.Jacobian(func)
    #     v = TgLh(gval) @ gdotatq.value
    #     return GroupTangentVector(h.L(q), v)
    
    def eval_left_lifted_action(self, g: RepGroupElement, h_config: RepGroupElement, h_dot):
        # ThLg @ h_dot
        return TangentVector(config = g.left_action(h_config), value = self.left_lifted_action(g, h_config) @ h_dot)

    def eval_right_lifted_action(self, h: RepGroupElement, g_config: RepGroupElement, h_dot: RepGroupElement):
        # TgRh @ h_dot
        return TangentVector(config = h.right_action(g_config), value = self.right_lifted_action(h, g_config) @ h_dot)

    def __repr__(self) -> str:
        return "Lie Group"

class LieGroupElement(RepGroupElement):
    def __init__(self, group, value, left_lifted_action, right_lifted_action):
        if isinstance(value, np.ndarray) and value.shape == group.identity.shape:
            save_value = value
        else:
            save_value = group.representation(value)
        super().__init__(group, save_value)
        self.left_lifted_action = left_lifted_action
        self.right_lifted_action = right_lifted_action

    def ad(self, gcircright: RepGroupElement):
        """Lifted adjoint action"""
        # return self.inverted_element.right_lifted_action(self.left_lifted_action(gcircright))
        g = self
        ginv = g.inverted_element
        TgRginv = self.right_lifted_action(ginv, g)
        TeLg = self.left_lifted_action(g, self.group.identity_element())
        return self.group.element(value=TgRginv @ TeLg @ gcircright.derepresentation)

    def ad_inv(self, gcircleft: RepGroupElement):
        """Lifted adjoint inv action"""
        g = self
        ginv = g.inverted_element
        TgLginv = self.left_lifted_action(ginv, g)
        TeRg = self.right_lifted_action(g, self.group.identity_element())
        return self.group.element(TgLginv @ TeRg @ gcircleft.derepresentation)



