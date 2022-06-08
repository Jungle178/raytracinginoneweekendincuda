#pragma once

#include "ray.h"
#include "aarect.h"
#include "hitable_list.h"

class box : public hitable {
public:
    __device__ box() {}
    __device__ box(const vec3& p0, const vec3& p1, material* ptr);
    __device__ ~box() {
        cudaFree(d_list);
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

    __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
        output_box = aabb(box_min, box_max);
        return true;
    }

public:
    vec3 box_min;
    vec3 box_max;
    hitable_list sides;
    hitable** d_list;
};

__device__ box::box(const vec3& p0, const vec3& p1, material* ptr) {
    box_min = p0;
    box_max = p1;
    //sides = new hitable_list();
    //hitable** d_list;
    cudaMalloc((void**)&d_list, 6 * sizeof(hitable*));
    int i = 0;
    d_list[i++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    d_list[i++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);
    d_list[i++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    d_list[i++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);
    d_list[i++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
    d_list[i++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);
    sides = hitable_list(d_list, 6);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return sides.hit(r, t_min, t_max, rec);
}