#!/usr/bin/env python

# import necessary modules
import rospy
import tf
from kuka_arm.srv import CalculateIK, CalculateIKResponse
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from sympy import symbols, cos, sin, pi, Matrix, sqrt, acos, atan2

def compute_inverse_kinematics(req):
    rospy.loginfo("Received %s end-effector poses from the plan" % len(req.poses))
    if len(req.poses) < 1:
        rospy.logwarn("No valid poses received")
        return -1
    else:
        # Define symbols for joint variables, link lengths, offsets, and twists
        q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
        a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
        d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
        alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')

        # Define Modified DH parameters
        dh_params = {
            alpha0: 0,      a0: 0,      d1: 0.75,   q1: q1,
            alpha1: -pi/2., a1: 0.35,   d2: 0,      q2: -pi/2. + q2,
            alpha2: 0,      a2: 1.25,   d3: 0,      q3: q3,
            alpha3: -pi/2., a3: -0.054, d4: 1.5,    q4: q4,
            alpha4: pi/2,   a4: 0,      d5: 0,      q5: q5,
            alpha5: -pi/2., a5: 0,      d6: 0,      q6: q6,
            alpha6: 0,      a6: 0,      d7: 0.303,  q7: 0
        }

        # Function to compute the DH transformation matrix
        def create_dh_matrix(alpha, a, d, q):
            return Matrix([
                [cos(q), -sin(q), 0, a],
                [sin(q) * cos(alpha), cos(q) * cos(alpha), -sin(alpha), -sin(alpha) * d],
                [sin(q) * sin(alpha), cos(q) * sin(alpha), cos(alpha), cos(alpha) * d],
                [0, 0, 0, 1]
            ])

        # Compute individual transformation matrices
        T0_1 = create_dh_matrix(alpha0, a0, d1, q1).subs(dh_params)
        T1_2 = create_dh_matrix(alpha1, a1, d2, q2).subs(dh_params)
        T2_3 = create_dh_matrix(alpha2, a2, d3, q3).subs(dh_params)
        T3_4 = create_dh_matrix(alpha3, a3, d4, q4).subs(dh_params)
        T4_5 = create_dh_matrix(alpha4, a4, d5, q5).subs(dh_params)
        T5_6 = create_dh_matrix(alpha5, a5, d6, q6).subs(dh_params)
        T6_7 = create_dh_matrix(alpha6, a6, d7, q7).subs(dh_params)
        T0_7 = T0_1 * T1_2 * T2_3 * T3_4 * T4_5 * T5_6 * T6_7

        # Define rotation matrices for roll, pitch, and yaw
        r, p, y = symbols('r p y')
        rot_x = Matrix([
            [1, 0, 0],
            [0, cos(r), -sin(r)],
            [0, sin(r), cos(r)]
        ])
        rot_y = Matrix([
            [cos(p), 0, sin(p)],
            [0, 1, 0],
            [-sin(p), 0, cos(p)]
        ])
        rot_z = Matrix([
            [cos(y), -sin(y), 0],
            [sin(y), cos(y), 0],
            [0, 0, 1]
        ])

        ROT_EE = rot_z * rot_y * rot_x
        rot_err = rot_z.subs(y, pi) * rot_y.subs(p, -pi/2)
        ROT_EE = ROT_EE * rot_err

        # Initialize response
        joint_trajectory_list = []
        for i in range(len(req.poses)):
            joint_trajectory_point = JointTrajectoryPoint()

            # Extract end-effector position and orientation
            px, py, pz = req.poses[i].position.x, req.poses[i].position.y, req.poses[i].position.z
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([
                req.poses[i].orientation.x,
                req.poses[i].orientation.y,
                req.poses[i].orientation.z,
                req.poses[i].orientation.w
            ])

            # Substitute the roll, pitch, and yaw values into the rotation matrix
            ROT_EE_eval = ROT_EE.subs({'r': roll, 'p': pitch, 'y': yaw})
            EE_pos = Matrix([[px], [py], [pz]])
            wrist_center = EE_pos - 0.303 * ROT_EE_eval[:, 2]

            theta1 = atan2(wrist_center[1], wrist_center[0])
            side_a, side_b = 1.501, sqrt((sqrt(wrist_center[0]**2 + wrist_center[1]**2) - 0.35)**2 + (wrist_center[2] - 0.75)**2)
            side_c = 1.25

            angle_a = acos((side_b**2 + side_c**2 - side_a**2) / (2 * side_b * side_c))
            angle_b = acos((side_a**2 + side_c**2 - side_b**2) / (2 * side_a * side_c))
            angle_c = acos((side_a**2 + side_b**2 - side_c**2) / (2 * side_a * side_b))

            theta2 = pi/2 - angle_a - atan2(wrist_center[2] - 0.75, sqrt(wrist_center[0]**2 + wrist_center[1]**2) - 0.35)
            theta3 = pi/2 - (angle_b + 0.036)

            # Evaluate R0_3 with the calculated thetas
            R0_3_eval = (T0_1[:3, :3] * T1_2[:3, :3] * T2_3[:3, :3]).evalf(subs={q1: theta1, q2: theta2, q3: theta3})
            R3_6 = R0_3_eval.T * ROT_EE_eval

            theta4 = atan2(R3_6[2, 2], -R3_6[0, 2])
            theta5 = atan2(sqrt(R3_6[0, 2]**2 + R3_6[2, 2]**2), R3_6[1, 2])
            theta6 = atan2(-R3_6[1, 1], R3_6[1, 0])

            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)

        rospy.loginfo("Length of Joint Trajectory List: %s" % len(joint_trajectory_list))
        return CalculateIKResponse(joint_trajectory_list)

def ik_service():
    rospy.init_node('IK_server')
    service = rospy.Service('calculate_ik', CalculateIK, compute_inverse_kinematics)
    rospy.loginfo("Ready to receive an IK request")
    rospy.spin()

if __name__ == "__main__":
    ik_service()
