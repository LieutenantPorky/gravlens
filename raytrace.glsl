#version 430
#define RAY_NUM %ray_num%
#define GRAV_SOURCE %grav_source%
#define MASS %mass%
#define DT %dt%
// %token% will be replaced by the corresponding value when parsed by python
struct Ray { //define how data is packaged in a Ray
  dvec4 position;
  dvec4 direction;

};

layout (local_size_x = RAY_NUM) in;

layout(binding=0) buffer rays // define the memory buffer to be bound to the CPU buffer
{
    Ray rays[];
} Ray_Buff;

void main() {
  dvec4 normal;
  double d; // distance to source
  dvec4 gradient; // local gradient of the gravitational potential - vec4 for compatibility with position, but ignore last comp
  dvec4 cgrad; // perpendicular component of the gradient
  dvec4 dgrad; // parallel component of the gradient

  Ray current_ray;
  uint ray_index = gl_GlobalInvocationID.x; // get the index of the current thread
  current_ray = Ray_Buff.rays[ray_index]; // we want this thread to work on the appropriate ray

  // first, calculate distance and normal to source
  d = distance(current_ray.position.xyz, GRAV_SOURCE);
  normal = dvec4(normalize(GRAV_SOURCE - current_ray.position.xyz), 0);

  // grad (1/r) = 1/r^2 r^
  gradient = double(MASS/pow(float(d),2.0)) * normal;
  dgrad = dot(current_ray.direction, gradient) * current_ray.direction;
  cgrad = gradient - dgrad;

  // calculate the Shapiro delay derivative (as a vec4, and using the last component as time)
  dvec4 dl = vec4(0,0,0,1) + vec4(0,0,0,1) * dot(dgrad,current_ray.direction) * 2 * DT;

  // calculate new direction
  // current_ray.direction = dvec4(normalize(current_ray.direction.xyz + MASS * length(cross(normal.xyz, current_ray.direction.xyz)) * normal.xyz/pow(float(d),3.0)), 0);

  current_ray.direction = normalize(current_ray.direction + 2 * cgrad * DT);

  // Update the position and direction of the ray
  current_ray.position = current_ray.position + DT * (current_ray.direction + dl);
  Ray_Buff.rays[ray_index].direction = current_ray.direction;
  Ray_Buff.rays[ray_index].position = current_ray.position;


}
