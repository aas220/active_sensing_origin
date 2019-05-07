//
// Created by tipakorng on 8/2/17.
//

#ifndef ACTIVE_SENSING_CONTINUOUS_MATH_UTILS_H
#define ACTIVE_SENSING_CONTINUOUS_MATH_UTILS_H

#include <Eigen/Dense>

namespace math
{

    inline double gaussianPdf(const Eigen::VectorXd &x, const Eigen::VectorXd &mean, const Eigen::MatrixXd &cov)
    {
        double n = (x - mean).transpose() * cov.inverse() * (x - mean);
        return exp(-0.5 * n) / sqrt(pow(2 * M_PI, cov.rows()) * cov.determinant());
    }

    inline Eigen::Matrix3d so3rotation(const Eigen::Vector3d &axis, double angle)
    {
        Eigen::Matrix3d so3matrix;
        double axis_norm = axis.norm();

        if (axis_norm * std::abs(angle) < std::numeric_limits<double>::epsilon())
        {
            so3matrix = Eigen::Matrix3d::Identity();
        } else
        {
            Eigen::Matrix3d axis_matrix;
            axis_matrix << 0, -axis(2), axis(1),
                    axis(2), 0, -axis(0),
                    -axis(1), axis(0), 0;
            so3matrix = Eigen::Matrix3d::Identity() +
                        axis_matrix / axis_norm * sin(axis_norm * angle) +
                        axis_matrix * axis_matrix / pow(axis_norm, 2) * (1 - cos(axis_norm * angle));
        }

        return so3matrix;
    }

    inline Eigen::Matrix3d so3rotation(const Eigen::Vector3d &axis)
    {
        return so3rotation(axis, 1);
    }

    template<typename T>
    inline int sgn(T val)
    {
        return (T(0) < val) - (val < T(0));
    }

}

#endif //ACTIVE_SENSING_CONTINUOUS_MATH_UTILS_H
