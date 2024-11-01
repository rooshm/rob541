#! /usr/bin/python3
import os
import sys

parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
sys.path.append(parent_dir)
from copy import deepcopy
from typing import List, Union
from geomotion import (
    utilityfunctions as ut,
    representationliegroup as rplg,
    rigidbody as rb,
)
from Assignments import simplekinematicchain as kc
import numpy as np
from matplotlib import pyplot as plt
from plotter import plot_vector_field

np.set_printoptions(precision=3, suppress=True)

# Set the group as SE2 from rigidbody
G = rb.SE2


class DiffKinematicChain(kc.KinematicChain):

    def __init__(
        self,
        links: List[rplg.RepresentationLieGroupElement],
        joint_axes: List[rplg.RepresentationLieGroupTangentVector],
    ):
        """Initialize a kinematic chain augmented with Jacobian methods"""

        # Call the constructor for the base KinematicChain class
        super().__init__(links, joint_axes)

        # Create placeholders for the last-calculated Jacobian value and the index it was requested for
        self.last_J: Union[None, np.ndarray] = None
        self.last_index = None
        self.last_spatial = None

    def Jacobian_Ad_inv(
        self, link_index, output_frame="body"  # Link number (with 1 as the first link)
    ):  # options are world, body, spatial
        """Calculate the Jacobian by using the Adjoint_inverse to transfer velocities from the joints to the links"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints
        J = np.zeros((len(G.identity_element().value), len(self.joint_axes)))

        ########
        # Populate the Jacobian matrix by finding the transform from each joint before the chosen link to the
        # end of the link, and using its Adjoint-inverse to transform the joint axis to the body frame of
        # the selected link, and then transform this velocity to the world, or spatial coordinates if requested

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry
        self.link_positions_with_base = [l for l in self.link_positions]
        self.link_positions_with_base.insert(0, G.identity_element())
        #! useful to get just once
        selected_link: rplg.RepresentationLieGroupElement = self.link_positions[
            link_index - 1
        ]

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):

            # create a transform g_rel that describes the position of the selected link relative the jth joint (which
            # is at the (j-1)th location in link_positions_with_base
            g_rel = self.link_positions_with_base[j - 1].inverse * selected_link

            # use the Adjoint-inverse of this relative transformation to map the jth joint axis ( (j-1)th entry)
            # out to the end of the selected link in the link's body frame
            J_joint = g_rel.Ad_inv(self.joint_axes[j - 1])

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates
            #! use left lifted action to get gdot
            if output_frame == "world":
                result = selected_link.TL(J_joint)

            # If the output_frame input is 'spatial', use the adjoint of the link position to map the axis back to
            # the identity
            #! body to spatial conversion at link
            elif output_frame == "spatial":
                result = selected_link.Ad(J_joint)
            elif output_frame == "body":
                result = J_joint
            else:
                raise ValueError(
                    f"Only body, spatial and world frame supported! you gave {output_frame}!"
                )

            # Insert the value of J_joint into the (j-1)th index of J
            J[:, j - 1] = np.asarray(result.value).flatten()

        # Store J and the last requested index
        self.last_index = link_index
        self.last_J = deepcopy(J)

        return J

    def Jacobian_Ad(
        self, link_index, output_frame="body"  # Link number (with 1 as the first link)
    ):  # options are world, body, spatial
        """Calculate the Jacobian by using the Adjoint to transfer velocities from the joints to the origin"""

        # Construct Jacobian matrix J as an ndarray of zeros with as many rows as the group has dimensions,
        # and as many columns as there are joints
        J = np.zeros((len(G.identity_element().value), len(self.joint_axes)))

        ########
        # Populate the Jacobian matrix by finding the position of each joint in the world (which is the same as the
        # position of the previous link), and using its Adjoint to send the axis into spatial coordinates

        # Make a list named link_positions_with_base that is the same as self.link_positions, but additionally has an
        # identity element inserted before the first entry
        self.link_positions_with_base = [l for l in self.link_positions]
        self.link_positions_with_base.insert(0, G.identity_element())

        selected_link: rplg.RepresentationLieGroupElement = self.link_positions[
            link_index - 1
        ]

        # Loop from 1 to link_index (i.e., range(1, link_index+1) )
        for j in range(1, link_index + 1):

            # use the Adjoint of the position of this joint to map its joint axis ( (j-1)th entry)
            # back to the identity of the group
            joint_pos = self.link_positions_with_base[j - 1]
            J_joint = joint_pos.Ad(self.joint_axes[j - 1])

            # If the output_frame input is 'world', map the axis from the Lie algebra out to world coordinates
            if output_frame == "world":
                result = selected_link.TR(J_joint)

            # If the output_frame input is 'body', use the adjoint-inverse of the link position to map the axis back to
            # the identity
            elif output_frame == "body":
                result = selected_link.Ad_inv(J_joint)
            elif output_frame == "spatial":
                result = J_joint
            else:
                raise ValueError(
                    f"Only body, spatial and world frame supported! you gave {output_frame}!"
                )

            # Insert the value of J_joint into the (j-1)th index of J
            J[:, j - 1] = np.asarray(result.value).flatten()

        # Store J and the last requested index
        self.last_J = deepcopy(J)
        self.last_index = link_index

        return J

    def draw_Jacobian(self, ax: plt.Axes):
        """Draw the components of the last-requested Jacobian"""

        # Get the location of the last-requested link, and use ut.column to make into a numpy column array
        last_link = ut.column(self.link_positions[self.last_index - 1])

        # Use np.tile to make a matrix in which each column is the coordinates of the link end
        configs = np.tile(last_link, len(self.last_J[0]))

        # Use ax.quiver to plot a set of arrows at the selected link end, (remembering to use only the xy components
        # and not the theta component)
        x_configs = configs[0, :].flatten()
        y_configs = configs[1, :].flatten()
        # jacobian as many rows as the group has dimensions, and as many columns as there are joints
        j_x = self.last_J[0, :]
        j_y = self.last_J[1, :]

        # draw axes to ensure full arrow is visible
        max_config_x = np.max(x_configs)
        min_config_x = np.min(x_configs)
        max_config_y = np.max(y_configs)
        min_config_y = np.min(y_configs)
        x_spacing = 1
        y_spacing = 2
        ax.plot(
            [0, 0],
            [min(min_config_y - y_spacing, 0), max_config_y + y_spacing],
            "--k",
            alpha=0.0,
        )
        ax.plot(
            [min(min_config_x - x_spacing, 0), max_config_x + x_spacing],
            [0, 0],
            "--k",
            alpha=0.0,
        )
        # plot arrows
        ax.quiver(
            x_configs,
            y_configs,
            j_x,
            j_y,
            angles="xy",
            scale_units="xy",
            scale=4.0,
            width=0.006,
            color="red",
        )
        ax.grid()
        ax.set_aspect('equal', adjustable='datalim')


if __name__ == "__main__":
    # Create a list of three links, all extending in the x direction with different lengths
    links = [G.element([3, 0, 0]), G.element([2, 0, 0]), G.element([1, 0, 0])]

    # Create a list of three joint axes, all in the rotational direction
    joint_axes = [G.Lie_alg_vector([0, 0, 1])] * 3

    # Create a kinematic chain
    kc = DiffKinematicChain(links, joint_axes)
    config1 = [0.25 * np.pi, -0.5 * np.pi, 0.75 * np.pi]
    config2 = [-0.25 * np.pi, 0.5 * np.pi, 0.75 * np.pi]
    config3 = [0.0, 0.0, 0.0]
    configs = [config1, config2, config3]

    # Create a plotting axis
    fig, ax = plt.subplots(2, 3, layout="constrained")
    fig.set_size_inches(25, 16)
    fig.suptitle("Jacobian at [1 1 1] rotational velocities")

    for i, ax_config in enumerate(ax[0]):

        # Draw the chain at config i
        kc.set_configuration(configs[i])
        kc.draw(ax_config)

        # per link jacobian draw
        for j in range(1, len(links) + 1):
            J_Ad = kc.Jacobian_Ad_inv(j, "world")
            print(J_Ad)
            kc.draw_Jacobian(ax_config)
        ax_config: plt.Axes
        ax_config.set_title(f"Jacobian from body desc. {np.round(configs[i], 2)}")

    print("spatial construction")
    for i, ax_config in enumerate(ax[1]):

        # Draw the chain at config i
        kc.set_configuration(configs[i])
        kc.draw(ax_config)

        # per link jacobian draw
        for j in range(1, len(links) + 1):
            J_Ad = kc.Jacobian_Ad(j, "world")
            print(J_Ad)
            kc.draw_Jacobian(ax_config)
        ax_config: plt.Axes
        ax_config.set_title(f"Jacobian from spatial desc. {np.round(configs[i], 2)}")
    print(J_Ad)

    plt.savefig("jacobian.png")
    # plt.show()
