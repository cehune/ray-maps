# Ray Maps for Global Illumination with Segments
From what I understand, this is the only implementation of "Ray Maps for Global Illumination" by V. Havran submitted to Eurographics 2005. 
This is a non-official implementation, taken to understand the paper on a deeper level and use for a segment based approach. 

The idea with ray maps builds off of photon maps. Photon maps add energy contribution at a radius of influence around an impact point. However, ray maps take it a step further, aware that rays may pass very close to a surface point, but not quite hit nearby. It's particularly powerful for small objects, where you have few impacts on the object, but possible many rays that graze by. 

This uses a similar estimator, but rather than a distance on photon impact to surface point, the norm is the distance to the ray passing through a hemisphere disk that is tangent to the surface point. 

This implementation does NOT implement LRU caching of the tree or a full render, but rather just many unit tests and a qualitative comparison by running the ./bias build command, and then the bias visualization python file in data_visualization.
