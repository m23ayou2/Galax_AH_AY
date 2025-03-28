
#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}










constexpr float softeningSquared = 1.0f;
constexpr float G = 10.0f;
constexpr float dt = 0.1f;
constexpr float dt2 = 2.0f;


void Model_CPU_fast::step_v2()
{
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        accelerationsx[i] = 0.0f;
        accelerationsy[i] = 0.0f;
        accelerationsz[i] = 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        const float xi = particles.x[i];
        const float yi = particles.y[i];
        const float zi = particles.z[i];

        #pragma omp simd reduction(+:ax,ay,az)
        for (int j = 0; j < n_particles; j++) {
            if (i == j) continue;
            
            const float dx = particles.x[j] - xi;
            const float dy = particles.y[j] - yi;
            const float dz = particles.z[j] - zi;

            const float distSq = dx*dx + dy*dy + dz*dz + softeningSquared;
            const float invDist = 1.0f / std::sqrt(distSq);
            const float invDistCube = invDist * invDist * invDist * G;

            ax += dx * invDistCube * initstate.masses[j];
            ay += dy * invDistCube * initstate.masses[j];
            az += dz * invDistCube * initstate.masses[j];
        }

        accelerationsx[i] = ax;
        accelerationsy[i] = ay;
        accelerationsz[i] = az;
    }

    #pragma omp parallel for simd
    for (int i = 0; i < n_particles; i++) {
        velocitiesx[i] += accelerationsx[i] * dt2;
        velocitiesy[i] += accelerationsy[i] * dt2;
        velocitiesz[i] += accelerationsz[i] * dt2;
        
        particles.x[i] += velocitiesx[i] * dt;
        particles.y[i] += velocitiesy[i] * dt;
        particles.z[i] += velocitiesz[i] * dt;
    }
}












void Model_CPU_fast::step()
{
    #pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        accelerationsx[i] = 0.0f;
        accelerationsy[i] = 0.0f;
        accelerationsz[i] = 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
              b_type raccx_i = b_type::load_unaligned(&accelerationsx[i]);
              b_type raccy_i = b_type::load_unaligned(&accelerationsy[i]);
              b_type raccz_i = b_type::load_unaligned(&accelerationsz[i]);

        for (int j = 0; j < n_particles; j++)
        {
            const b_type rposx_j = b_type(particles.x[j]);
            const b_type rposy_j = b_type(particles.y[j]);
            const b_type rposz_j = b_type(particles.z[j]);

            const b_type diffx = rposx_j - rposx_i;
            const b_type diffy = rposy_j - rposy_i;
            const b_type diffz = rposz_j - rposz_i;

            const b_type dij_sq = diffx * diffx + diffy * diffy + diffz * diffz;
            const b_type dij_ = xs::rsqrt(dij_sq) ;
            const b_type dij = xs::select(dij_sq < b_type(1.0), b_type(10.0), b_type(10.0) * (dij_ * dij_* dij_));

            raccx_i += diffx * dij * b_type(initstate.masses[j]);
            raccy_i += diffy * dij * b_type(initstate.masses[j]);
            raccz_i += diffz * dij * b_type(initstate.masses[j]);
        }

        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);
    }

    for (int i = 0; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * dt2;
        velocitiesy[i] += accelerationsy[i] * dt2;
        velocitiesz[i] += accelerationsz[i] * dt2;
        particles.x[i] += velocitiesx[i] * dt;
        particles.y[i] += velocitiesy[i] * dt;
        particles.z[i] += velocitiesz[i] * dt;
    }
}


  
    
#endif // GALAX_MODEL_CPU_FAST