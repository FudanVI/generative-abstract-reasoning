import create_polygon as poly_data
import create_circle as circle_data
import create_complex_polygon as complex_poly_data
import create_complex_circle as complex_circle_data
import create_position_polygon as position_poly_data
import create_position_circle as position_circle_data


def get_drawer(name):
    if name.split('_')[0] == 'circle':
        drawer = circle_data.InstanceDrawerContinue(64)
        attr = circle_data.get_attributions()
        bound = circle_data.get_bounds()
        return drawer, attr, bound
    elif name.split('_')[0] == 'triangle':
        drawer = poly_data.InstanceDrawer(3, 64)
        attr = poly_data.get_attributions()
        bound = poly_data.get_bounds(3)
        return drawer, attr, bound
    elif name.split('_')[0] == 'square':
        drawer = poly_data.InstanceDrawer(4, 64)
        attr = poly_data.get_attributions()
        bound = poly_data.get_bounds(4)
        return drawer, attr, bound
    elif name.split('_')[0] == 'complex' and name.split('_')[1] == 'circle':
        drawer = complex_circle_data.InstanceDrawerComplexCircle(64)
        attr = complex_circle_data.get_attributions()
        bound = complex_circle_data.get_bounds()
        return drawer, attr, bound
    elif name.split('_')[0] == 'complex' and name.split('_')[1] == 'polygon':
        drawer = complex_poly_data.InstanceDrawerComplexPolygon(3, 3, 1.0, 0.3, 64)
        attr = complex_poly_data.get_attributions()
        bound = complex_poly_data.get_bounds(3)
        return drawer, attr, bound
    elif name.split('_')[0] == 'position' and name.split('_')[1] == 'circle':
        drawer = position_circle_data.InstanceDrawerContinue(64)
        attr = position_circle_data.get_attributions()
        bound = position_circle_data.get_bounds()
        return drawer, attr, bound
    elif name.split('_')[0] == 'position' and name.split('_')[1] == 'triangle':
        drawer = position_poly_data.InstanceDrawer(3, 64)
        attr = position_poly_data.get_attributions()
        bound = position_poly_data.get_bounds(3)
        return drawer, attr, bound
    elif name.split('_')[0] == 'position' and name.split('_')[1] == 'square':
        drawer = position_poly_data.InstanceDrawer(4, 64)
        attr = position_poly_data.get_attributions()
        bound = position_poly_data.get_bounds(4)
        return drawer, attr, bound
    else:
        assert False
