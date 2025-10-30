import pygame
import moderngl
import numpy as np
import glm
import random
from PIL import Image
from loadModelUsingAssimp_V3 import create3DAssimpObject

pygame.init()

# Window setup
width = 600
height = 600
screen = pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.RESIZABLE)
pygame.display.set_caption(title="Assignment 10: Cody Taylor")
gl = moderngl.get_context()

# Vertex shader
vertex_shader_code = '''
    #version 460 core
    layout (location=0) in vec3 position;
    layout (location=1) in vec3 normal;
    layout (location=2) in vec2 uv;

    layout(binding = 0, std430) readonly buffer InstanceData {
        mat4 instanceMatrix[];
    };
    
    uniform mat4 view;
    uniform mat4 perspective;
    
    out vec2 f_uv;
    out vec3 f_normal;
    out vec3 f_position;
    
    void main()
    {
        mat4 model = instanceMatrix[gl_InstanceID];
        vec4 in_position = model * vec4(position, 1.0);
        f_position = in_position.xyz;
        gl_Position = perspective * view * in_position;
        mat3 normalMatrix = mat3(transpose(inverse(model)));
        f_normal = normalize(normalMatrix * normal);
        f_uv = uv;
    }
'''

# Fragment shader
fragment_shader_code = '''
    #version 460 core
    
    in vec2 f_uv;
    in vec3 f_normal;
    in vec3 f_position;
    out vec4 outColor;
    
    uniform sampler2D map;
    
    uniform float ambientLight;
    uniform float shininess;
    uniform vec3 specularColor;
    uniform vec3 k_diffuse;
    
    uniform vec4 lightDir;
    uniform vec3 eyePoint;

    
    void main()
    {
        vec3 light;
        if (lightDir.w > 0)
        {
            light = normalize(lightDir.xyz - f_position);
        }
        else 
        {
            light = normalize(lightDir.xyz);
        }

        vec3 normals = normalize(f_normal);
        vec3 materialColor = texture(map, f_uv).rgb;
        
        vec3 color = vec3(0);

        if (dot(light, normals) > 0)
        {
            vec3 sightVector = normalize(f_position - eyePoint);
            vec3 H = normalize (light + sightVector);

            float diffuseLighting = clamp(dot(light, normals), 0, 1);
            vec3 specularlyReflectedColor = specularColor * pow(clamp(dot(H, normals), 0, 1), shininess);
        
            float lighting = ambientLight + diffuseLighting;

            color = k_diffuse * (materialColor * lighting) + specularlyReflectedColor;
        }
        
        outColor = vec4(color, 1);
    }
'''
program = gl.program(vertex_shader=vertex_shader_code, fragment_shader=fragment_shader_code)

# Takes in object file. Arranges the vertices and texture coordinates.

object = create3DAssimpObject("./mario_obj/scene.gltf", verbose=False, normalFlag=True, textureFlag=True, tangentFlag=False)

bound = object.bound
object.createRenderableAndSampler(program=program)
renderables = object.renderables
samplers = object.samplers

# Sets the look at point to the origin and the up vector.
lookAtPoint = glm.vec3(0,0,0)
upVector = glm.vec3(0.0, 1.0, 0.0)

floor_vertex_shader_code = '''
    #version 460 core
    layout (location=0) in vec3 position;
    layout (location=1) in vec3 normal;
    layout (location=2) in vec2 uv;
    
    uniform mat4 view;
    uniform mat4 perspective;
    uniform mat4 model;
    
    out vec2 f_uv;
    out vec3 f_normal;
    out vec3 f_position;
    
    void main()
    {
        vec4 in_position = model * vec4(position, 1.0);
        f_position = in_position.xyz;
        gl_Position = perspective * view * in_position;
        mat3 normalMatrix = mat3(transpose(inverse(model)));
        f_normal = normalize(normalMatrix * normal);
        f_uv = uv;
    }
'''

floor_fragment_shader =  '''
    #version 460 core
    
    in vec2 f_uv;
    in vec3 f_normal;
    in vec3 f_position;
    out vec4 outColor;
    
    uniform sampler2D map;
    
    uniform float ambientLight;
    
    uniform vec4 lightDir;
    uniform vec3 eyePoint;

    
    void main()
    {
        vec3 light;
        if (lightDir.w > 0)
        {
            light = normalize(lightDir.xyz - f_position);
        }
        else 
        {
            light = normalize(lightDir.xyz);
        }

        vec3 normals = normalize(f_normal);
        vec3 materialColor = texture(map, f_uv).rgb;
        
        vec3 color = vec3(0);

        if (dot(light, normals) > 0)
        {
            vec3 sightVector = normalize(f_position - eyePoint);
            vec3 H = normalize(light + sightVector);

            float diffuseLighting = clamp(dot(light, normals), 0, 1);
        
            float lighting = ambientLight + diffuseLighting;

            color = (materialColor * lighting);
        }
        
        outColor = vec4(color, 1);
    }
'''
floor_program = gl.program(vertex_shader=floor_vertex_shader_code, fragment_shader=floor_fragment_shader)

def create_floor(size, texture_path):
    vertices = []
    normals = []
    uvs = []
    indices = []

    vertices = [[-size/2, 0.0, -size/2],
                [size/2, 0.0, -size/2],
                [size/2, 0.0, size/2],
                [-size/2, 0.0, size/2]]

    normals = [[0.0, 1.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0, 1.0, 0.0]]
    
    uvs = [[0.0, 0.0],
           [1.0, 0.0],
           [1.0, 1.0],
           [0.0, 1.0]]

    indices = [[0, 1, 2],
               [0, 2, 3]]

    floor_data, floor_normals, floor_uvs, floor_indices = np.array(vertices, dtype='f4'), np.array(normals, dtype='f4'), np.array(uvs, dtype='f4'), np.array(indices, dtype='i4')
    floor_data_buffer = gl.buffer(floor_data)
    floor_normal_buffer = gl.buffer(floor_normals)
    floor_uvs_buffer = gl.buffer(floor_uvs)
    floor_indices_buffer = gl.buffer(floor_indices)

    floorvba = gl.vertex_array(floor_program,
                    [(floor_data_buffer, "3f", "position"), 
                     (floor_normal_buffer, "3f", "normal"),
                     (floor_uvs_buffer, "2f", "uv")],
                    index_buffer=floor_indices_buffer)
    
    image = Image.open(texture_path).transpose(Image.FLIP_TOP_BOTTOM).convert("RGB")
    texture = gl.texture(image.size, 3, image.tobytes())
    texture.build_mipmaps()

    return floorvba, texture

def RenderFloor(program, vba, texture, vertexArgs, fragmentArgs):
    program["view"].write(vertexArgs[0])
    program["perspective"].write(vertexArgs[1])

    model_matrix = glm.rotate(glm.mat4(1.0), glm.radians(0.0), glm.vec3(1.0, 0.0, 0.0))
    program["model"].write(np.array(model_matrix, dtype='f4').T.tobytes())

    program["ambientLight"].value = fragmentArgs[0]
    program["lightDir"].value = fragmentArgs[1]

    texture.use(location=1)
    program["map"].value = 1
    vba.render(moderngl.TRIANGLES)

floor_size = 10
floorVBA, floorTex = create_floor(floor_size, "./textures/floor-wood.jpg")

def MakeTransformations(size):
    transformations = []
    model_size = bound.boundingBox[1] - bound.boundingBox[0]
    max_model_size = np.max(model_size)
    scale_factor = 0.75 / max_model_size
    middle_index = (size - 1) / 2
    for x in range(size):
        for z in range(size):
            xPos = (-middle_index + x) + random.uniform(-0.5, 0.5)
            zPos = (-middle_index + z) + random.uniform(-0.5, 0.5)

            scale = glm.scale(glm.mat4(1.0), glm.vec3(scale_factor))
            upRotation = glm.rotate(glm.mat4(1.0), glm.radians(-90), glm.vec3(1,0,0))
            horizontalRotation = glm.rotate(glm.mat4(1.0), glm.radians(random.uniform(-30,30)), glm.vec3(0,1,0))
            rotation = horizontalRotation * upRotation
            yPos = (0)
            translation = glm.translate(glm.mat4(1.0), glm.vec3(xPos, yPos, zPos))
            transformations.append(translation * rotation * scale)

    transformationArray = np.array([np.array(i, dtype='f4').T.flatten() for i in transformations], dtype='f4').tobytes()
    transformationBuffer = gl.buffer(transformationArray)
    transformationBuffer.bind_to_storage_buffer(0)

    return len(transformations)



transformation_count = MakeTransformations(floor_size)

# Sets the camera distance and the viewing angle.
camera_y_angle = 90
camera_distance_global = 25
camera_distance = camera_distance_global
camera_rotation_angle = 0.0
camera_rotation_speed = 0.5 

# Sets the light source distance and angle.
light_y_angle = 45
light_distance = 3.0 * bound.radius
light_rotation_angle = 0.0
light_rotation_interval = 1
rotateLight = 0

# Sets the fov, near plane, and far plane of the camera.
fov = 45/2
near_plane = bound.radius * 0.1  
far_plane = bound.radius * 10.0 

gl.enable(moderngl.DEPTH_TEST)
gl.disable(moderngl.CULL_FACE)

# Main loop
running = True

debug = False
mouse_camera_control = False
last_mouse_pos = [0, 0]
cur_mouse_pos = [0, 0]

lightSetting = 0
spinning = True


clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False
        if event.type == pygame.VIDEORESIZE:
            width, height = event.size
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_l):
            lightSetting = 1
            print("Point Lighting")
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_i):
            lightSetting = 0
            print("Directional Lighting")
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_p):
            spinning = not spinning
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT):
            rotateLight = -1
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT):
            rotateLight = 1
        if (event.type == pygame.KEYUP and (event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT)):
            rotateLight = 0
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_d):
            debug = not debug
            if (debug):
                print("Debug On")
            else:
                print("Debug Off")
                camera_distance = camera_distance_global
                camera_y_angle = 90

        if (debug == True):
            if event.type == pygame.MOUSEWHEEL:
                camera_distance += -(event.y * 0.5)
                if (camera_distance < 0):
                    camera_distance = 0
            if event.type == pygame.MOUSEBUTTONDOWN:
                last_mouse_pos = event.pos
                cur_mouse_pos = last_mouse_pos
                mouse_camera_control = True
                print(last_mouse_pos)
            if (event.type == pygame.MOUSEMOTION) and (mouse_camera_control):
                cur_mouse_pos = event.pos
                if (camera_y_angle > 175):
                    camera_y_angle = 175
                elif (camera_y_angle < 5):
                    camera_y_angle = 5
            if event.type == pygame.MOUSEBUTTONUP:
                mouse_camera_control = False

            
    
    gl.clear(0.3, 0.3, 0.4)
    gl.clear(depth=True)
    
    # Finds the viewing point of the camera based off of it's current rotation.
    cam_angle_rad = np.radians(camera_rotation_angle)
    cam_theta = np.radians(camera_y_angle) 
    cam_phi = cam_angle_rad 

    camerax = camera_distance * np.sin(cam_theta) * np.sin(cam_phi)
    if (debug):
        cameray = camera_distance * np.cos(cam_theta)
    else:
        cameray = 10
    cameraz = camera_distance * np.sin(cam_theta) * np.cos(cam_phi)

    eyePoint = glm.vec3(camerax, cameray, cameraz)

    light_angle_rad = np.radians(light_rotation_angle)
    light_theta = np.radians(light_y_angle)
    light_phi = light_angle_rad
    lightx = light_distance * np.sin(light_theta) * np.sin(light_phi)
    lighty = bound.center.y
    lightz = light_distance * np.sin(light_theta) * np.cos(light_phi)

    lightPoint = glm.vec3(lightx, lighty, lightz)

    # Creates the viewing matrix and perspective matrix to send to the shader.
    view_matrix = glm.lookAt(eyePoint, lookAtPoint, upVector)
    aspect_ratio = width / height
    perspective_matrix = glm.perspective(glm.radians(fov), aspect_ratio, near_plane, far_plane)
    
    model_matrix = glm.rotate(glm.mat4(1.0), glm.radians(-90.0), glm.vec3(1.0, 0.0, 0.0))

    vertexArgs = [np.array(view_matrix, dtype='f4').T.tobytes(), np.array(perspective_matrix, dtype='f4').T.tobytes(), np.array(model_matrix, dtype='f4').T.tobytes()]
    fragmentArgs = [0.05, (lightPoint.x, lightPoint.y, lightPoint.z, lightSetting)]


    program['view'].write(vertexArgs[0])
    program['perspective'].write(vertexArgs[1])
    
    program['ambientLight'].value = fragmentArgs[0]
    program['lightDir'].value = fragmentArgs[1]
    program['specularColor'].value = (1, 1, 1)

    RenderFloor(floor_program, floorVBA, floorTex, vertexArgs, fragmentArgs)
    object.render(nInstances=transformation_count)
    pygame.display.flip()
    
    if (rotateLight != 0):
        light_rotation_angle += rotateLight * light_rotation_interval

    if (debug == False and spinning):
        camera_rotation_angle += camera_rotation_speed
        if camera_rotation_angle >= 360.0:
            camera_rotation_angle = 0
    elif (mouse_camera_control):
        camera_rotation_angle += -(cur_mouse_pos[0] - last_mouse_pos[0])
        camera_y_angle += -(cur_mouse_pos[1] - last_mouse_pos[1])
        last_mouse_pos = cur_mouse_pos

    
    clock.tick(60)

pygame.quit()