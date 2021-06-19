# Racing Car using GA/PSO algorithm
This project consists of creating a toy car racing enviroment, where we can train our car to complete a race track, by training it on multiple tracks using the Genetic Algorithm(GA) and PSO(Particle Swarm Optimizer) algorithm. This project was done as a part of the course Neural Networks and Fuzzy Logic at BITS Pilani. 

## Code Files

[Tracks](tracks) - Directory that contains the tracks used for training. Sample tracks have been added. 

[Visualization](visualization) - Directory where the GIFs of how the race car performs on the tracks are stored. 

[Weights](weights) - Directory where the weights are stored after training. 

[base_car.py](base_car.py) - Contains the Car class that implements most of the functionalities.

[car_gapso_imp.py](car_gapso_imp.py) - Contains template for GA/PSO training. It also contains the mechanism through which the car moves. 

[train_ga.py](train_ga.py) - Training script for training the car based on Genetic Algorithm.

[train_pso.py](train_pso.py) - Training script for training the car based on PSO algorithm.

[car_custom_imp.py](car_gapso_imp.py) - Contains template for GA/PSO training. It also contains the mechanism through which the car moves.

[custom_train.py](train_ga.py) - Loads the custom car implemented in [car_custom_imp.py](car_gapso_imp.py) and finds the fitness on different tracks. 

## How to Run

Optionally, create a virtual environment on your system and open it. 

To run the project, first clone the repository by typing the command in git bash.
```
git clone https://github.com/imshreyshah/Racing-Car-GA-PSO.git
```

Alternatively, you can download the code as .zip and extract the files.

Shift to the cloned directory
```
cd Racing-Car-GA-PSO
```

To install the requirements, run the following command:
```
pip install -r requirements.txt
```

Run the training script for the algorithm you want to use. 

To run the GA algorithm, use the following command: 
```
python train_ga.py
```

The visualizations would be stored in the visualization directory. 

The sample outputs would look like: -

![Sample output 1](https://github.com/imshreyshah/Racing-Car-GA-PSO/blob/main/Sample/template_1_gatrack1.gif)

![Sample output 2](https://github.com/imshreyshah/Racing-Car-GA-PSO/blob/main/Sample/template_1_gatrack2.gif)

![Sample output 3](https://github.com/imshreyshah/Racing-Car-GA-PSO/blob/main/Sample/template_1_gatrack3.gif)

## Acknowledgement
Thanks to the Instructor and the Teacher's Assistants for implementing many functionalities of this assignment. 
