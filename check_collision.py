import numpy as np
from hjelpefunksjonar import norm
from constants import vel_limit

def check_all_collisions(particle, data, ribs):
    nearest_collision = None
    field_width = ribs[1].get_rib_middle()[0] - ribs[0].get_rib_middle()[0]
    data_avstand = (data[0] - particle.init_position[0]) % field_width
    data[0] = particle.init_position[0] + data_avstand
    for rib in ribs:
        collision_info = check_collision(particle, data, rib)
        collision_info['rib'] = rib
        if nearest_collision is None:
            nearest_collision = collision_info
        if collision_info['collision_depth'] > nearest_collision['collision_depth']:
            nearest_collision = collision_info
        if nearest_collision['is_collision'] or nearest_collision['is_resting_contact'] or nearest_collision['is_leaving']:
            break
    return nearest_collision

def check_collision(particle, data, rib):
    """
    Sjekkar kollisjonar mellom ein partikkel med ein posisjon og ei ribbe. Brukar metoden i 
    Tanaya, Chen, Pavleas, Sung (2017) Building a 2D Game Physics Engine: Using HTML5 and JavaScript, 
    "The Rectangle Circle Collision Project"

    Parameters
    ----------
    data : tuple
        Ein tuple (x,y,u,v) som gjev koordinatane og farten til senter av partikkelen.
    rib : Rib
        Den aktuelle ribba, altså eit rektangel.

    Returns
    -------
    dict
        Ein dict med fylgjande data: (boolean, collisionInfo, rib). 
    """
    assert data.shape == (4,)
    position = data[0:2]        
    inside = True
    bestDistance = -99999
    nearestEdge = 0

    # collisionInfo = (-1,np.array([-1,-1]), np.array([-1,-1]), -1)
    # collisionInfo = {}
    
    #Step A - compute nearest edge
    vertices = rib.vertices
    normals = rib.normals
    
    for i in range(len(vertices)):
        v = position - vertices[i]
        projection = np.dot(v, normals[i][:,0])
        if (projection > 0):
            # if the center of circle is outside of rectangle
            bestDistance = projection
            nearestEdge = i
            inside = False
            break
        
        if (projection > bestDistance):
            # If the center of the circle is inside the rectangle
            bestDistance = projection
            nearestEdge = i
            
    
    if (not inside):            
        #  Step B1: If center is in Region R1
        # the center of circle is in corner region of mVertex[nearestEdge]

        # //v1 is from left vertex of face to center of circle
        # //v2 is from left vertex of face to right vertex of face
        v1 = position - vertices[nearestEdge]
        v2 = vertices[(nearestEdge + 1) % 4] - vertices[nearestEdge]
        
        dot = np.dot(v1, v2)
        
        if (dot < 0): #region R1
            dis = np.sqrt(v1.dot(v1))
            
            if (dis > particle.radius):
                return {'is_collision':False, 'collision_depth': particle.radius - dis, 'is_resting_contact':False, 'is_leaving':False}#, 'rib':rib}
                # (False, collisionInfo, rib) # må vel endra til (bool, depth, normal, start)
            
            normal = norm(np.reshape(v1,(2,1)))
            
            radiusVec = normal*particle.radius*(-1)
            
            # sender informasjon til collisioninfo:                    
            collision_info = dict(collision_depth=particle.radius - dis, rib_normal=normal, particle_collision_point = position + radiusVec, inside = inside)
                        
        else:
            # //the center of circle is in corner region of mVertex[nearestEdge+1]
    
            #         //v1 is from right vertex of face to center of circle 
            #         //v2 is from right vertex of face to left vertex of face
            v1 = position - vertices[(nearestEdge +1) % 4]
            v2 = (-1) * v2
            dot = v1.dot(v2)
            
            if (dot < 0):
                dis = np.sqrt(v1.dot(v1))
                                   
                # //compare the distance with radium to decide collision
        
                if (dis > particle.radius):
                    return {'is_collision':False, 'collision_depth': particle.radius - dis, 'is_resting_contact':False, 'is_leaving':False}#, 'rib':rib}
                    # (False, collisionInfo, rib) # må vel endra til (bool, depth, normal, start)

                normal = norm(np.reshape(v1,(2,1)))
                radiusVec = normal * particle.radius*(-1)
                
                collision_info = dict(collision_depth=particle.radius - dis, rib_normal = normal, particle_collision_point = position + radiusVec, inside = inside)
            else:
                #//the center of circle is in face region of face[nearestEdge]
                if (bestDistance < particle.radius):
                    radiusVec = normals[nearestEdge][:,0] * particle.radius
                    collision_info = dict(collision_depth = particle.radius - bestDistance, rib_normal = normals[nearestEdge], particle_collision_point = position - radiusVec, inside = inside)
                else:
                    return dict(is_collision =  False, collision_depth = particle.radius - bestDistance, is_resting_contact = False, is_leaving = False, inside = inside)
    else:
        # the center of circle is inside of rectangle
        radiusVec = normals[nearestEdge][:,0] * particle.radius

        return dict(is_collision = True, is_resting_contact = False, is_leaving = False, rib = rib, collision_depth = particle.radius - bestDistance, rib_normal = normals[nearestEdge], particle_collision_point = position - radiusVec, inside = inside)
        # Måtte laga denne returen så han ikkje byrja å rekna ut relativ fart når partikkelen uansett er midt inne i ribba.

    # Rekna ut relativ fart i retning av normalkomponenten, jamfør Baraff (2001) formel 8-3.
    n = collision_info['rib_normal'][:,0]
    v = np.array(data[2:])
    v_rel = np.dot(n,v)
    collision_info['relative_velocity'] = v_rel
    collision_info['rib'] = rib
    # collision_info['closest_rib_normal'] = normals[nearestEdge]

    if (abs(v_rel) < vel_limit and round(np.dot(n, normals[nearestEdge][:,0]),3)==1.0 ):
        collision_info['is_resting_contact'] = True
    else:
        collision_info['is_resting_contact'] = False

    if (v_rel < -vel_limit): # Sjekk om partikkelen er på veg vekk frå veggen. Negativ v_rel er på veg mot vegg, positiv er på veg ut av vegg.
        collision_info['is_collision'] = True
    else:
        collision_info['is_collision'] = False

    if v_rel > vel_limit and particle.resting:
        collision_info['is_leaving'] = True
    else:
        collision_info['is_leaving'] = False
    
    return collision_info