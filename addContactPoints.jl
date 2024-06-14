
# # Add visualizers to feet so we know where we're adding contact points
# radius = 0.01
# z = -0.083
# left_foot = findbody(robot, "LEFT_FOOT_LINK")
# left_foot_frame = default_frame(left_foot)
# setelement!(mvis, Point3D(left_foot_frame, -0.057, -0.047, z), radius, "pt1")
# setelement!(mvis, Point3D(left_foot_frame, -0.057, 0.047, z), radius, "pt2")
# setelement!(mvis, Point3D(left_foot_frame, 0.165, -0.028, z), radius, "pt3")
# setelement!(mvis, Point3D(left_foot_frame, 0.165, 0.028, z), radius, "pt4")
# right_foot = findbody(robot, "RIGHT_FOOT_LINK")
# right_foot_frame = default_frame(right_foot)
# setelement!(mvis, Point3D(right_foot_frame, -0.057, 0.047, z), radius, "pt1")
# setelement!(mvis, Point3D(right_foot_frame, -0.057, -0.047, z), radius, "pt2")
# setelement!(mvis, Point3D(right_foot_frame, 0.165, 0.028, z), radius, "pt3")
# setelement!(mvis, Point3D(right_foot_frame, 0.165, -0.028, z), radius, "pt4")

# # Add contact points to feet
# contactmodel = SoftContactModel(hunt_crossley_hertz(k = 500e3), ViscoelasticCoulombModel(0.8, 20e3, 100.))
# add_contact_point!(left_foot, ContactPoint(Point3D(left_foot_frame, -0.057, -0.047, z), contactmodel))
# add_contact_point!(left_foot, ContactPoint(Point3D(left_foot_frame, -0.057, 0.047, z), contactmodel))
# add_contact_point!(left_foot, ContactPoint(Point3D(left_foot_frame, 0.165, -0.028, z), contactmodel))
# add_contact_point!(left_foot, ContactPoint(Point3D(left_foot_frame, 0.165, 0.028, z), contactmodel))
# add_contact_point!(right_foot, ContactPoint(Point3D(right_foot_frame, -0.057, 0.047, z), contactmodel))
# add_contact_point!(right_foot, ContactPoint(Point3D(right_foot_frame, -0.057, -0.047, z), contactmodel))
# add_contact_point!(right_foot, ContactPoint(Point3D(right_foot_frame, 0.165, 0.028, z), contactmodel))
# add_contact_point!(right_foot, ContactPoint(Point3D(right_foot_frame, 0.165, -0.028, z), contactmodel))