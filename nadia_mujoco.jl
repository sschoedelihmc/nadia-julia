using Pkg; Pkg.activate(@__DIR__);
using MuJoCo; init_visualiser()
using LinearAlgebra

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