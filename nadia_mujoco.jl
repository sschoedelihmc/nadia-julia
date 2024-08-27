using Pkg; Pkg.activate(@__DIR__);
using QuadrupedControl

include(joinpath(@__DIR__, "nadia_robot.jl"))
model = Nadia()
let
    global intf
    close(intf)
    intf = init_mujoco_interface(model)
    data = init_data(model, intf, preferred_monitor=3)
    
    intf.d.qpos[3] = 1.2
end

foot_id = 11
torso_id = 14
unsafe_string(MuJoCo.mj_id2name(m, MuJoCo.mjOBJ_MESH, foot_id - 1))
unsafe_string(MuJoCo.mj_id2name(m, MuJoCo.mjOBJ_MESH, torso_id - 1))
foot_start = m.mesh_graphadr[foot_id] + 1
foot_end = m.mesh_graphadr[torso_id]
hull_data = m.mesh_graph[foot_start:foot_end]
numvert = hull_data[1]
numface = hull_data[2]
vert_edgeadr = hull_data[2 .+ (1:numvert)]
vert_globalid = hull_data[2 + numvert .+ (1:numvert)]
edge_localid = hull_data[2 + 2*numvert .+ (1:numvert + 3*numface)]
face_globalid = hull_data[2 + 3*numvert + 3*numface .+ (1:3*numface)]

vert_adr = m.mesh_vertadr[foot_id]
q = m.mesh_quat[foot_id, :]
verts = [quat_to_rot(q)*m.mesh_vert[vert_adr - 1 + v, :] + m.mesh_pos[foot_id, :] for v in vert_globalid]

verts = [quat_to_rot(q)*m.mesh_vert[vert_adr - 1 + k, :] + m.mesh_pos[foot_id, :] for k in 1:m.mesh_vertnum[foot_id]]
verts = [v for v in verts if v[3] < -0.0879 && (v[1] < -0.0715 || v[1] > 0.1775)]


plot([v[1] for v in verts], [v[2] for v in verts])



plot(m.mesh_vert[1:3:end], m.mesh_vert[2:3:end], m.mesh_vert[3:3:end])

using Plots

let
    put!(c, true) 
    wait(rendertask)
    global m, d, c, p, rendertask
    path = joinpath(@__DIR__, "nadia_V17_description/mujoco/nadiaV17.simpleKnees_scene.xml")
    m = load_model(path)
    d = MuJoCo.init_data(m)
    c = Channel()
    p, rendertask = visualise!(m, d, channel = c, preferred_monitor=3)
    @lock p.lock begin d.qpos[3] = 1.2; MuJoCo.mj_forward(m, d); end
end

let 
end

test = MuJoCo.mj_parseXML(urdf_path, C_NULL, C_NULL, C_NULL)

load_model

M = [-0.00787035 -1.667e-4 -0.00075945 
     -1.667e-4 0.00727481 -0.000988837
     -0.00075945  -0.000988837 -0.00182144]

abstract type TestingFunc end
abstract type Blah <: TestingFunc end
struct B2 <: Blah end
abstract type B3{TestingFunc} end
struct B5{T} <: B3 end
function test2(real::T) where T<:TestingFunc
    println(T)
    println(typeof(real))
    println("here 3")
end