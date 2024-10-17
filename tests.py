import numpy as np
from convenience import OpEnum, OpGen
from group import Group, DirectProduct, SemiDirectProduct, SE2, GroupElement
from repgroup import RepGroup, RepGroupElement

if __name__ == '__main__':
    # test 1 affine addition
    op = OpGen(OpEnum.AFFINE_ADD, {'dim': 2})
    g = Group(op.operation, op.inverse, op.identity, op.params)
    g1 = g.element(np.array([1, 2]))
    g2 = g.element(np.array([3, 4]))
    assert np.isclose(g1.left_action(g2).value, np.array([4, 6])).all()
    assert np.isclose(g1.right_action(g2).value, np.array([4, 6])).all()
    
    # test 2 scalar multiplication
    op = OpGen(OpEnum.SCALAR_PRODUCT)
    g = Group(op.operation, op.inverse, op.identity)
    g1 = g.element(2)
    g2 = g.element(3)
    assert np.isclose(g1.left_action(g2).value, 6)
    assert np.isclose(g1.right_action(g2).value, 6)

    # test 2 modular addition
    op = OpGen(OpEnum.MODULAR_ADD, {"phi": 360})
    g = Group(op.operation, op.inverse, op.identity, op.params)
    g1 = g.element(5)
    g2 = g.element(360) 
    assert np.isclose(g1.left_action(g2).value, 5)
    assert np.isclose(g1.right_action(g2).value, 5)

    # test 3 direct product
    scale = OpGen(OpEnum.SCALAR_PRODUCT)
    mg = Group(scale.operation, scale.inverse, scale.identity)
    shift = OpGen(OpEnum.SCALAR_ADD)
    ad = Group(shift.operation, shift.inverse, shift.identity)
    gh = DirectProduct(mg, ad)
    g1 = gh.element([3., -1.])
    g2 = gh.element([1/2, 3/2])
    g1g2 = g1.left_action(g2)
    g2g1 = g1.right_action(g2)
    assert np.isclose(g1g2.value, np.array([1.5, 0.5])).all()
    assert np.isclose(g2g1.value, np.array([1.5, 0.5])).all()
    # commutivity of group
    hg = DirectProduct(mg, ad)
    h1 = hg.element([3., -1.])
    h2 = hg.element([1/2, 3/2])
    h1h2 = h1.left_action(h2)
    h2h1 = h1.right_action(h2)
    assert np.isclose(h1h2.value, np.array([1.5, 0.5])).all()
    assert np.isclose(h2h1.value, np.array([1.5, 0.5])).all()

    # test 4 semi direct product
    scale = OpGen(OpEnum.SCALAR_PRODUCT)
    mg = Group(scale.operation, scale.inverse, scale.identity)
    shift = OpGen(OpEnum.SCALAR_ADD)
    ad = Group(shift.operation, shift.inverse, shift.identity)
    gh = SemiDirectProduct(mg, ad)
    g1 = gh.element([3., -1.])
    g2 = gh.element([1/2, 3/2])
    g1g2_l = g1.left_action(g2).value
    g1g2_r = g2.right_action(g1).value
    assert np.isclose(g1g2_l, np.array([1.5, 3.5])).all()
    assert np.isclose(g1g2_r, np.array([1.5, 3.5])).all()
    # non commutivity
    g2g1_l = g2.left_action(g1).value
    g2g1_r = g1.right_action(g2).value
    assert np.isclose(g2g1_l, np.array([1.5, 1])).all()
    assert np.isclose(g2g1_r, np.array([1.5, 1])).all()
    

    # part 1 SE2
    se2 = SE2()
    # test figure 1.29
    g = se2.element([2., 1.,  np.pi/2])
    h = se2.element([0., -1.,  -np.pi/4])
    gh_l = g.left_action(h).value
    gh_r = h.right_action(g).value
    assert np.isclose(gh_l, np.array([3., 1., np.pi/4])).all()
    assert np.isclose(gh_r, np.array([3., 1., np.pi/4])).all()
    # non commutivity
    hg_l = h.left_action(g).value
    hg_r = g.right_action(h).value
    assert np.isclose(hg_r, hg_l).all()
    assert not np.isclose(hg_l, np.array([3., 1., np.pi/4])).all()

    # representation test 1 affine addition
    aff = OpGen(OpEnum.AFFINE_ADD, {'dim': 2})
    op = RepGroup(aff.representation, aff.derepresentation, np.eye(3), params={'dim': 2})
    g1 = op.element(np.array([1, 2]))
    g2 = op.element(np.array([3, 4]))
    assert np.isclose(g1.left_action(g2).value, op.element(np.array([4, 6])).value).all()
    assert np.isclose(g1.right_action(g2).value, op.element(np.array([4, 6])).value).all()
    
   