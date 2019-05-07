function p = potential(x, attCoeff, repCoeff, radius, threshold)
    
    p = attractive_potential(x, attCoeff) + ...
        repulsive_potential(x, repCoeff, radius, threshold);

end

function p = attractive_potential(x, attCoeff)

    p = 0.5 * attCoeff * x(1:2, 1)' * x(1:2, 1);

end

function p = repulsive_potential(x, repCoeff, radius, threshold)

    numObstacles = size(x, 1) / 2 - 1;
    max_potential = 10;
    p = 0;
    
    for i = 1 : 1 : numObstacles
        obstacleIndices = 2 * i + 1 : 2 * i + 2;
        relativePosition = x(1:2, 1) - x(obstacleIndices, 1);
        distance = norm(relativePosition, 2) - radius;
        
        if distance < threshold 
            if distance > radius
                p = p + min(max_potential, 0.5 * repCoeff * (1/distance - 1/threshold)^2);
            else
                p = p + max_potential;
            end
        end
    end

end
