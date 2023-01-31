import math

DISABLE_INTERSECT = False
def div(a, b):
    """
    Handles division with infinity
    a: dividend
    b: divisor
    """
    if abs(a) == math.inf and abs(b) == math.inf:
        raise ValueError("Cannot divide infinity by infinity")
    try:
        return a/b
    except ZeroDivisionError:
        return math.copysign(math.inf, a)


def get_normal(x1, y1, x2, y2):
    """
    y distance for unit change in x perpendicular to a line
    """
    return div(x2 - x1, y2 - y1)


def get_intersect(x1, y1, m1, x2, y2, m2):
    """
    Returns the intersection of two lines as a tuple (x, y)
    """
    # The formula for intersection is
    # ((b1×c2 − b2×c1)/(a1×b2 − a2×b1), (c1×a2 − c2×a1)/(a1×b2 − a2×b1))
    # for ax + by + c = 0
    # if y = mx + c
    # b = -1, a = m and c= y1 - m1.x1
    c1 = y1 - m1 * x1
    c2 = y2 - m2 * x2
    if m2 == m1:
        return x1, y1
    elif m1 == math.inf:
        x = x1
    elif m2 == math.inf:
        x = x2
    else:
        x = div(c1 - c2, (m2-m1))
    if m1 == math.inf:
        y = m2 * x + c2
    elif m2 == math.inf:
        y = m1 * x + c1
    else:
        y = div((c1 * m2 - c2 * m1), (m2 - m1))    
    return x, y


def is_sharp_angle(m1, m2):
    """
    Checks whether the angle between two lines is too small
    to use their intersect
    """
    angle = (m1 - m2) / (1 + m1*m2)
    return DISABLE_INTERSECT or angle>0 and angle< 0.577 # tan(30)


def line_to_poly(points_x, points_y, thickness):
    """
    Convert a given line to a polygon
    """
    # First half of the polygon points
    m1_x = []
    m1_y = []
    # Second half of the polygon points
    m2_x = []
    m2_y = []
    s_x = points_x
    s_y = points_y
    last_grad = 0
    half_th = thickness/2
    for i in range(1, len(points_x)):
        # The normal of the current line segment
        normal = get_normal(s_x[i-1], s_y[i-1], s_x[i], s_y[i])
        # Instead of using cosines, we use the hypotenuse of the normal and a unit displacement in the x direction
        # The lw is then measured along this hypotenuse
        scale = math.hypot(1, normal)
        dx = half_th * div(1, scale)
        dy = half_th * (div(normal, scale) if normal != scale else 1)
        if (s_y[i-1] > s_y[i]):
            dx, dy = -dx, -dy

        # Expand line using normal to form a rectangle with 4 points
        point_1 = s_x[i-1]+dx, s_y[i-1]-dy
        point_2 = s_x[i-1]-dx, s_y[i-1]+dy
        point_3 = s_x[i]+dx, s_y[i]-dy
        point_4 = s_x[i]-dx, s_y[i]+dy
        

        gradient = div(1, normal)
        if i > 1 and not is_sharp_angle(gradient, last_grad):
            # Smooth out the corners of each join by taking the intersect
            # For very sharp corners, this will cause artifacts
            intersect_1 = get_intersect(
                m1_x[-1], m1_y[-1], last_grad, point_1[0], point_1[1], gradient)
            intersect_2 = get_intersect(
                m2_x[-1], m2_y[-1], last_grad, point_2[0], point_2[1], gradient)
            m1_x[-1] = intersect_1[0]
            m2_x[-1] = intersect_2[0]
            m1_y[-1] = intersect_1[1]
            m2_y[-1] = intersect_2[1]
        else:
            m1_x.append(point_1[0])
            m2_x.append(point_2[0])
            m1_y.append(point_1[1])
            m2_y.append(point_2[1])
        last_grad = gradient
        m1_x.append(point_3[0])
        m2_x.append(point_4[0])
        m1_y.append(point_3[1])
        m2_y.append(point_4[1])
    return m1_x+m2_x[-1::-1], m1_y+m2_y[-1::-1]
