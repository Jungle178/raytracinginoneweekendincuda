#ifndef RAYH
#define RAYH
#include "vec3.h"


__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}
__device__ float random_float(float min, float max, curandState* local_rand_state) {
    return min + (max - min) * curand_uniform(local_rand_state);
}
__device__ int random_int(int min, int max, curandState* local_rand_state) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_float(min, max + 1, local_rand_state));
}
__device__ inline float degrees_to_radians(float degrees) {
    return degrees * 3.14159265358979323846 / 180.0;
}
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; tm = 0; }
        __device__ ray(const vec3& origin, const vec3& direction, float time)
        { 
            A = origin; B = direction; tm = time;
        }
        __device__ vec3 origin() const       { return A; }
        __device__ vec3 direction() const    { return B; }
        __device__ double time() const { return tm; }
        __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

        vec3 A;
        vec3 B;
        float tm;
};

#endif
