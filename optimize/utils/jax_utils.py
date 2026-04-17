import jax
import jax.numpy as jnp


def pytree_norm(pytree):
    """
    Computes the L2 norm of a pytree
    """
    squares = jax.tree_util.tree_map(lambda x: jnp.sum(x**2), pytree)
    total_square = jax.tree.reduce(lambda leaf_1, leaf_2: leaf_1 + leaf_2, squares)
    return jnp.sqrt(total_square)


jprint = lambda *args: [jax.debug.print("{var}", var=arg) for arg in args]


# Calculate cosine similarity between 2 pytree vectors
def cosine_similarity(pytree1, pytree2):
    vec1 = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(pytree1)])
    vec2 = jnp.concatenate([jnp.ravel(x) for x in jax.tree.leaves(pytree2)])
    dot_product = jnp.dot(vec1, vec2)
    norm1 = jnp.linalg.norm(vec1)
    norm2 = jnp.linalg.norm(vec2)
    denominator = norm1 * norm2
    cosine_sim = jnp.where(denominator > 1e-8, dot_product / denominator, 0.0)
    return cosine_sim
