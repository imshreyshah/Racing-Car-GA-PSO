'''
A template to see how template_car_2 is performing
No training is required since the car is completely hardcoded
'''

from car_custom_imp import MyCar
import numpy as np

track1 = 'tracks/sample_track_1.jpg'
track2 = 'tracks/sample_track_2.jpg'
track3 = 'tracks/sample_track_3.jpg'
track4 = 'tracks/sample_track_4.jpg'
track5 = 'tracks/sample_track_5.jpg'
track6 = 'tracks/sample_track_6.jpg'
track7 = 'tracks/sample_track_7.jpg'
track8 = 'tracks/sample_track_8.jpg'
track9 = 'tracks/sample_track_9.jpg'

car = MyCar(id=0, weights=np.zeros(6,)) # weights are dummy as we are not using them in our move function at all

# Visualize on different tracks
f1 = car.run(track1, save='template_2_track1.gif')
f2 = car.run(track2, save='template_2_track2.gif')
f3 = car.run(track3, save='template_2_track3.gif')
f4 = car.run(track4, save='template_2_track4.gif')
f5 = car.run(track5, save='template_2_track5.gif')
f6 = car.run(track6, save='template_2_track6.gif')
f7 = car.run(track7, save='template_2_track7.gif')
f8 = car.run(track8, save='template_2_track8.gif')
f9 = car.run(track9, save='template_2_track9.gif')
print(f'Overall fitness: {f1+f2+f3+f4+f5+f6+f7+f8+f9} = {f1} + {f2} + {f3} + {f4} + {f5}+ {f6}+ {f7}+ {f8}+ {f9}')

# save (dummy) weights
car.save(file='template_2_weights')
