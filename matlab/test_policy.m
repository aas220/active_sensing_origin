%
% This script 
%

%% Initialization

% Initialize model parameters.
numObstacles = 7;
stateSize = 2 * (numObstacles + 1);
dimensions = 2;
radius = 0.025;
obstacles = [0.50; -0.50; 
             0.50;  0.00;
             0.50;  0.50;
             0.25; -0.25;
             0.25;  0.25;
             0.75; -0.25;
             0.75;  0.25];

% Initialize planner parameters.
threshold = 0.3;
attCoeff = 1.0;
repCoeff = 1e-3;

%% Calculate potential field

x = -1:0.01:1;
y = -1:0.01:1;
potentials = zeros(length(y), length(x));

for i = 1 : 1 : length(x)
    
    for j = 1 : 1 : length(y)
        z = [x(i); y(j); obstacles];
        potentials(j, i) = potential(z, attCoeff, repCoeff, radius, threshold);
    end
    
end

%% Plot

surf(x, y, potentials)