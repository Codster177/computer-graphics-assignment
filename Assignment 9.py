import pygame
import moderngl
import numpy as np
import glm
from PIL import Image
from loadModelUsingAssimp_V2 import create3DAssimpObject

pygame.init()

# Window setup
width = 600
height = 600
screen = pygame.display.set_mode((width, height), flags=pygame.OPENGL | pygame.RESIZABLE)
pygame.display.set_caption(title="Assignment 9: Cody Taylor")
gl = moderngl.get_context()

# Vertex shader
vertex_shader_code = '''
    #version 460 core
    layout (location=0) in vec3 in_position;
    layout (location=1) in vec3 in_normal;
    layout (location=2) in vec2 in_uv;
    
    uniform mat4 view;
    uniform mat4 perspective;
    uniform mat4 model;
    
    out vec2 f_uv;
    out vec3 f_normal;
    out vec3 f_position;
    
    void main()
    {
        vec4 position = model * vec4(in_position, 1.0);
        f_position = position.xyz;
        gl_Position = perspective * view * position;
        mat3 normalMatrix = mat3(transpose(inverse(model)));
        f_normal = normalize(normalMatrix * in_normal);
        f_uv = in_uv;
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

            color = (materialColor * lighting) + specularlyReflectedColor;        
        }
        
        outColor = vec4(color, 1);
    }
'''
program = gl.program(vertex_shader=vertex_shader_code, fragment_shader=fragment_shader_code)

# Takes in object file. Arranges the vertices and texture coordinates.

object = create3DAssimpObject("./chair_table_class/scene.gltf", verbose=False, normalFlag=True, textureFlag=True, tangentFlag=False)

bound = object.bound
renderables = object.getRenderables(gl, program, "3f 3f 2f", ['in_position', 'in_normal', 'in_uv'])
samplers = object.getSamplers(gl)


# Sets the look at point to the origin and the up vector.
lookAtPoint = glm.vec3(bound.center)
upVector = glm.vec3(0.0, 1.0, 0.0)

# Sets the camera distance and the viewing angle.
camera_y_angle = 90
camera_distance = 2.0 * bound.radius
camera_rotation_angle = 0.0
camera_rotation_speed = 0.5 

# Sets the light source distance and angle.
light_y_angle = 45
light_distance = 3.0 * bound.radius
light_rotation_angle = 0.0
light_rotation_interval = 1
rotateLight = 0

# Sets the fov, near plane, and far plane of the camera.
fov = 45.0
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
                camera_distance = 2.0 * bound.radius
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
        cameray = bound.center.y
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

    program['view'].write(np.array(view_matrix, dtype='f4').T.tobytes())
    program['perspective'].write(np.array(perspective_matrix, dtype='f4').T.tobytes())
    program['model'].write(np.array(model_matrix, dtype='f4').T.tobytes())
    
    program['lightDir'].value = (lightPoint.x, lightPoint.y, lightPoint.z, lightSetting)
    program['ambientLight'].value = 0.05
    program['specularColor'].value = (1, 1, 1)

    object.render(program, renderables, samplers)
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