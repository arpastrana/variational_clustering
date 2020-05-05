import compas.geometry as cg


__all__ = ['align_vector']



def align_vector(vector_a, vector_b):
    if cg.dot_vectors(vector_a, vector_b) < 0:
        return cg.scale_vector(vector_a, -1.)
    else:
        return vector_a