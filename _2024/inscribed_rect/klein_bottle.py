
import numpy as np

TAU = 2 * np.pi
PI = np.pi
def klein_bottle_function(mode = "wikipedia"):
    if mode == "academic_paper":
        def klein_bottle(u, v, a=20, b=8, c=11/2, d=2/5):
            """
            Computes the x, y, z coordinates for a point on the Klein bottle
            based on the parametrization from section 3 of the paper.

            Parameters:
                t (float): Parameter along the directrix curve, in (0, 2π).
                theta (float): Angular parameter around the tube, in [0, 2π].
                a, b (float): Parameters controlling the directrix curve.
                c, d (float): Parameters controlling the tube radius.

            Returns:
                tuple: (x, y, z) coordinates of the point on the Klein bottle.
            """
            fake_u = 1e-7
            u *= TAU
            v *= TAU
            if u == 0:
                u = fake_u
            elif u == PI:
                u -= fake_u
            elif u == TAU:
                u -= fake_u
            # Directrix curve γ(t)
            gamma_x = a * (1 - np.cos(u))
            gamma_y = b * np.sin(u) * (1 - np.cos(u))

            # Radius function r(t)
            radius = c - d * (u - np.pi) * np.sqrt(max(0, u * (2 * np.pi - u)))

            # Tangent vector to γ(t)
            gamma_dx = a * np.sin(u)
            gamma_dy = b * (np.sin(u)**2 + np.cos(u) * (1 - np.cos(u)))
            norm = np.sqrt(gamma_dx**2 + gamma_dy**2)

            # Orthogonal vectors to γ'(t)
            J = (-gamma_dy / norm, gamma_dx / norm)  # Rotated tangent vector
            k = (0, 0, 1)  # z-axis unit vector

            # x, y, z coordinates of the tube around γ(t)
            x = gamma_x + radius * (np.cos(v) * J[0])
            y = gamma_y + radius * (np.cos(v) * J[1])
            z = radius * np.sin(v)

            return np.array([x, y, z])
        return klein_bottle
    elif mode == "skamkam":
        def surface_klein(u,v,
            h_top = 0.5,
            h_bottom = 1,
            w_right = 0.5,
            w_left = 0.3,
            r1 = 0.2,
            r2 = 0.35,
            r3 = 0.5):

            # difficulty with this parameterization:
            # consider what happens as the inner tube approaches the bottom:
            # r must be growing as u_z is at a standstill
            # u_z must be growing as u_x is at a standstill
            # so I don't think we can accomplish this with trig functions along
            # solution: we will use nested trig functions for u_x to make it doubly slow

            def jug_handle(t, h_top, h_bottom, w):
                # as t ranges from 0 to 1, this ranges from (0, h_top) to (0, -h_bottom) with a stop at (w, 0)
                # I think h_bottom should equal 2*h_top to have continuous second derivative, and w=h_top looks good
                if t < 0.5:
                    t0 = 2 * t
                    x = w * np.sin(1 / 2 * np.pi * t0)
                    z = h_top * np.cos(1/2 * np.pi * t0)
                else:
                    t1 = 2 * t - 1
                    # as tu goes from 0 to 1 so does tu_stretch, but it slows at the end
                    t_stretch = np.sin(np.pi / 2 * t1)
                    x = w * (0.5 * np.cos(np.pi * t_stretch) + 0.5)
                    z = h_bottom * np.cos(1/2 * np.pi * (t1+1))
                return x, z

            def partial_f(f, t):
                # numerical derivative of f with respect to t at t
                eps = 1e-6
                t_low = max(0, t-eps)
                t_high = min(1, t+eps)
                x_low, z_low = f(t_low)
                x_high, z_high = f(t_high)
                dx = (x_high - x_low) / (t_high - t_low)
                dz = (z_high - z_low) / (t_high - t_low)
                return dx, dz

            def cylinderify(f, t, v, r):
                x_u, z_u = f(t)
                y_u = 0

                dx, dz = partial_f(f, t)
                ds = np.sqrt(dx**2 + dz**2)

                x_v = r * np.cos(2*np.pi*v)
                y_v = r * np.sin(2*np.pi*v)

                x = x_u + x_v * (dz/ds)
                y = y_u + y_v
                z = z_u + x_v * (-dx/ds)
                return x, y, z
            if u < 0.5:
                # right hand jug handle
                t = 2*u
                f = lambda t0: jug_handle(t0, h_top, h_bottom, w_right)

                r = r1 if t < 0.5 else r1 + (1-np.cos(np.pi*(t-0.5)))*(r2 - r1)
                x, y, z = cylinderify(f, t, v, r)

            else:
                # left hand jug handle
                t = 2*u - 1
                f = lambda t0: jug_handle(1-t0, h_top, h_bottom, -w_left)

                r = r2 + np.sin(2*np.pi*t)*(r3-r2) if t < 0.5 else r2 + np.sin(np.pi*(t-0.5))*(r1-r2)
                # because the direction of f reverses at u=0.5, we need to do something to v so that
                # the seams line up correctly here. Note that dx/dt continues to be right-to-left, while
                # dz/dt switches from down to up, so to make the circles consistent we need to change
                # our process for v. I don't know why it is the way it is exactly
                x, y, z = cylinderify(f, t, (-v+0.5), r)
            return np.array([x, y, z])
        return surface_klein
    elif mode == "wikipedia":
        def klein_function(
            u, v, 
            a=2, b=1, c=15, scaling_factor=1, 
            p1=30, p2=90, p3=80, p4=60, p5=48,
            transformations=None
        ):
            """
            Generalized Klein bottle with parameterized constants.

            Parameters:
                u (float): First parametric variable, scaled to [0, 1].
                v (float): Second parametric variable, scaled to [0, 1].
                a (float): Scaling factor for x and z components.
                b (float): Scaling factor for the y component.
                c (float): Normalization constant to scale overall structure.
                scaling_factor (float): Additional scaling applied uniformly to x, y, and z.
                p1, p2, p3, p4, p5 (float): Adjustable constants for fine-tuning the equations.
                transformations (callable, optional): Function to apply additional transformations 
                                                    to the x, y, z coordinates.

            Returns:
                tuple: (x, y, z) coordinates.
            """
            PI = np.pi
            TAU = 2 * np.pi
            
            # Scale u and v
            u *= PI
            v *= TAU

            # Parametric equations with parameterized constants
            x = -a / c * np.cos(u) * (
                3 * np.cos(v) - p1 * np.sin(u) + p2 * np.cos(u)**4 * np.sin(u) -
                p4 * np.cos(u)**6 * np.sin(u) + 5 * np.cos(u) * np.cos(v) * np.sin(u)
            )
            y = -b / c * np.sin(u) * (
                3 * np.cos(v) - 3 * np.cos(u)**2 * np.cos(v) - p5 * np.cos(u)**4 * np.cos(v) +
                p5 * np.cos(u)**6 * np.cos(v) - p4 * np.sin(u) + 5 * np.cos(u) * np.cos(v) * np.sin(u) -
                5 * np.cos(u)**3 * np.cos(v) * np.sin(u) - p3 * np.cos(u)**5 * np.cos(v) * np.sin(u) +
                p3 * np.cos(u)**7 * np.cos(v) * np.sin(u)
            )
            z = a / c * (3 + 5 * np.cos(u) * np.sin(u)) * np.sin(v)
            
            # Apply additional scaling
            x *= scaling_factor
            y *= scaling_factor
            z *= scaling_factor
            
            # Apply optional transformations
            if transformations and callable(transformations):
                x, y, z = transformations(x, y, z)
            return np.array([x, y, z])
        return klein_function
    elif mode == "dickson":
        def klein_bottle_dickson(v, u, A=6, B=16, C=4):
            u *= TAU
            v *= TAU
            """
            Computes the (x, y, z) coordinates for Dickson's Klein bottle with parameters.

            Parameters:
                u (float): Parameter along the main curve, in [0, 2π].
                v (float): Parameter around the tube, in [0, 2π].
                A (float): Controls the main curve's size.
                B (float): Controls the overall height.
                C (float): Controls the tube radius.

            Returns:
                tuple: (x, y, z) coordinates of the point on the Klein bottle.
            """
            if 0 <= u <= np.pi:
                x = A * np.cos(u) * (1 + np.sin(u)) + C * (1 - 0.5 * np.cos(u)) * np.cos(u) * np.cos(v)
                y = B * np.sin(u) + C * (1 - 0.5 * np.cos(u)) * np.sin(u) * np.cos(v)
            else:
                x = A * np.cos(u) * (1 + np.sin(u)) + C * (1 - 0.5 * np.cos(u)) * np.cos(v + np.pi)
                y = B * np.sin(u)
            z = C * (1 - 0.5 * np.cos(u)) * np.sin(v)
            return np.array([x, y, z])
        return klein_bottle_dickson
    elif mode == "sinusoidal":
        def klein_bottle(u, v, a=3, b=1, mode = "cosine"):
            """
            Computes the (x, y, z) coordinates for a Klein bottle.

            Parameters:
                u (float): Parameter along the primary angular direction, in [0, 2π].
                v (float): Parameter along the secondary angular direction, in [0, 2π].
                a (float): Controls the size of the central tube.
                b (float): Controls the radius of the tube.

            Returns:
                tuple: (x, y, z) coordinates of the point on the Klein bottle.
            """
            u *= TAU
            v *= TAU
            if mode == "sinusoidal":
                x = (a + b * np.cos(v)) * np.cos(u)
                y = (a + b * np.cos(v)) * np.sin(u)
                z = b * np.sin(v) * np.sin(u)
            elif mode == "cosine":
                common_term = a + b * (np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v))
                x = common_term * np.cos(u)
                y = common_term * np.sin(u)
                z = b * (np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v))
            return np.array([x, y, z])
        return klein_bottle
    elif mode == "mathcurve":
        def klein_bottle(u, v, a=3, b=4, c=2):
            """
            Compute a Klein bottle based on the given parameters.
            
            Parameters:
                a (float): Parameter controlling the first dimension of the bottle.
                b (float): Parameter controlling the second dimension of the bottle.
                c (float): Parameter controlling the radial scaling of the tube.
                u_res (int): Resolution for the parameter u (default: 100).
                v_res (int): Resolution for the parameter v (default: 100).
                
            Returns:
                tuple: Meshgrid arrays for x, y, z coordinates of the Klein bottle.
            """
            # Define the range of u and v
            u *= 2 * np.pi
            v *= 2 * np.pi

            # Define r(u)
            r = c * (1 - np.cos(u) / 2)

            # First part (u: 0 to π)
            x1 = (a * (1 + np.sin(u)) + r * np.cos(v)) * np.cos(u)
            y1 = (b + r * np.cos(v)) * np.sin(u)

            # Second part (u: π to 2π)
            x2 = (a * (1 + np.sin(u)) * np.cos(u)) - (r * np.cos(v))
            y2 = b * np.sin(u)

            # Combine both parts
            x = x2 * (u > np.pi) + x1 * (u <= np.pi)
            y = y2 * (u > np.pi) + y1 * (u <= np.pi)
            z = r * np.sin(v)
            return np.array([x, y, z])
        return klein_bottle
    elif mode == "dumbbell":
        def klein_bottle_dumbbell(u, v):
            """
            Computes the (x, y, z) coordinates for the Klein bottle using
            the Dumbbell curve as the directrix.

            Parameters:
                t (float): Parameter along the directrix curve, in [0, π].
                theta (float): Angular parameter around the tube, in [0, 2π].

            Returns:
                tuple: (x, y, z) coordinates of the point on the Klein bottle.
            """
            fake_u = 1e-7
            u *= PI
            v *= 2*PI
            if u == 0:
                u = fake_u
            elif u == PI:
                u -= fake_u
            # Directrix curve α(t)
            alpha_x = 5 * np.sin(u)
            alpha_y = 2 * (np.sin(u)**2) * np.cos(u)

            # Radius function r(t)
            radius = 0.5 - (1/30) * (2 * u - np.pi) * np.sqrt(max(2 * u * (2 * np.pi - 2 * u), 0))

            # Tangent vector to α(t)
            alpha_dx = 5 * np.cos(u)
            alpha_dy = 2 * (2 * np.sin(u) * np.cos(u)**2 - np.sin(u)**3)
            norm = np.sqrt(alpha_dx**2 + alpha_dy**2)

            # Orthogonal vectors to α'(t)
            J = (-alpha_dy / norm, alpha_dx / norm)  # Rotated tangent vector

            # x, y, z coordinates of the tube around α(t)
            x = alpha_x + radius * (np.cos(v) * J[0])
            y = alpha_y + radius * (np.cos(v) * J[1])
            z = radius * np.sin(v)
            return np.array([x, y, z])
        return klein_bottle_dumbbell
        
    elif mode == "gemini":
        def klein_bottle(u, v):
            u *= TAU
            v *= TAU
            """
            Parametric equation for the Klein bottle.

            Args:
                u: First parameter.
                v: Second parameter.

            Returns:
                A tuple containing x, y, and z coordinates.
            """
            a = 6 * np.cos(u) * (1 + np.sin(u))
            b = 16 * np.sin(u)
            c = 5 * (1 - np.cos(u) / 2)

            # Handle the twist in the bottle's construction
            x = a + c * np.cos(v) * np.cos(u) 
            y = b + c * np.sin(v) * np.cos(u) 
            z = c * np.sin(v)
            x = (a + c * np.cos(v + np.pi)) * (u > np.pi) + x*(u <= np.pi)
            y = (b) * (u > np.pi) + y*(u <= np.pi)
            return np.array([x, y, z])
        return klein_bottle