#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"

class onb {
public:
    __device__ onb() {}

    __device__ inline vec3 operator[](int i) const { return axis[i]; }

    __device__ vec3 u() const { return axis[0]; }
    __device__ vec3 v() const { return axis[1]; }
    __device__ vec3 w() const { return axis[2]; }

    __device__ vec3 local(double a, double b, double c) const {
        return a * u() + b * v() + c * w();
    }

    __device__ vec3 local(const vec3& a) const {
        return a.x() * u() + a.y() * v() + a.z() * w();
    }

    __device__ void build_from_w(const vec3&);

public:
    vec3 axis[3];
};


__device__ void onb::build_from_w(const vec3& n) {
    axis[2] = unit_vector(n);
    vec3 a = (fabs(w().x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
    axis[1] = unit_vector(cross(w(), a));
    axis[0] = cross(w(), v());
}


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}

class material  {
    public:
        __device__ virtual vec3 emitted(float u, float v, const vec3& p) const {
            return vec3(0, 0, 0);
        }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
            return false;
        }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state, float& pdf) const {
            return false;
        }
        __device__ virtual float scattering_pdf(
            const ray& r_in, const hit_record& rec, const ray& scattered
        ) const {
            return 0;
        }
};

class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
             scattered = ray(rec.p, target-rec.p, r_in.time());
             attenuation = albedo;
             return true;
        }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state, float& pdf) const {
            //vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
            //scattered = ray(rec.p, target - rec.p, r_in.time());

            //onb uvw;
            //uvw.build_from_w(rec.normal);
            //auto scatter_direction = rec.normal + unit_vector(random_in_unit_sphere(local_rand_state));
            auto scatter_direction = rec.normal + unit_vector(random_in_unit_sphere(local_rand_state));
            scattered = ray(rec.p, unit_vector(scatter_direction), r_in.time());
            attenuation = albedo;
            pdf = dot(rec.normal, scattered.direction()) / M_PI;
            return true;
        }
        __device__ float scattering_pdf(
            const ray& r_in, const hit_record& rec, const ray& scattered
        ) const override {
            auto cosine = dot(rec.normal, unit_vector(scattered.direction()));
            return cosine < 0 ? 0 : cosine / M_PI;
        }

        vec3 albedo;
};

class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state), r_in.time());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
        vec3 albedo;
        float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered,
                         curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx*ref_idx*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected, r_in.time());
        else
            scattered = ray(rec.p, refracted, r_in.time());
        return true;
    }

    float ref_idx;
};
class diffuse_light : public material {
public:
    __device__ diffuse_light(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
        return false;
    }
    __device__ virtual vec3 emitted(float u, float v, const vec3& p) const override {
        return albedo;
    }
    vec3 albedo;
};

class isotropic : public material {
public:
    __device__ isotropic(vec3 c) : albedo(c) {}
    //isotropic(shared_ptr<texture> a) : albedo(a) {}

    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state
    ) const override {
        scattered = ray(rec.p, random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo;
        return true;
    }

public:
    vec3 albedo;
};
#endif
