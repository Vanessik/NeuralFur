source_paths_SMAL = {
    'panda':'/is/cluster/fast/bkabadayi/projects/SMALify/3d_panda5/Stage5.obj',
    'fox': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_fox5_1/Stage5.obj',
    'whiteTiger': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_whitetiger1_nolimb_posefirst_default_more/Stage5.obj',
    'cat': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_cat5/Stage5.obj',
    'beagle_dog':  '/is/cluster/fast/bkabadayi/projects/SMALify/3d_beagledog_5/Stage5.obj',
    'synth_tiger': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_syberian_aligned_LAST/Stage1_LAST_aligned_neus.obj'
    
}

source_paths_SMAL_furless = {
    'panda':'/is/cluster/fast/bkabadayi/projects/SMALify/3d_panda5_smaller/Stage5.obj',
     'whiteTiger': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_whitetiger1_nolimb_posefirst_default_more_smaller/Stage5.obj',
      'beagle_dog': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_beagledog_5_smaller/Stage5.obj',  
        'fox': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_fox5_1_smaller/Stage5.obj',
        'cat': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_cat5_smaller/Stage5.obj',
        'synth_tiger': '/is/cluster/fast/bkabadayi/projects/SMALify/3d_syberian_aligned_LAST_smaller/Stage1_smaller_aligned_furless.obj'
}



apply_scale_list = {
    'wolf': False,
    'beagle_dog': False,
    'whiteTiger': False,
    'fox': False,
    'panda': False,
    'cat': True,
    'bear': False,
    'synth_tiger':True
}

manes ={
    'wolf': False,
    'beagle_dog': False,
    'whiteTiger': True,
    'fox': False,
    'panda': False,
    'cat': False,
    'bear': False,
    'synth_tiger':True
}

postfixx = {
    'wolf': 'processed',
    'beagle_dog': 'processed2',
    'whiteTiger': 'processed2',
    'fox': 'processed',
    'panda': 'processed',
    'cat': 'processed',
    'bear': 'processed',
    'synth_tiger': 'processed2'
}


neus_dict = {
   'fox':'fox_07',
    'panda': 'panda_045',
    'whiteTiger': 'whiteTiger_035',
    'beagle_dog': 'beagle_dog_2_13',
    'whiteTiger': 'whiteTiger_035_2',
    'cat': 'cat',
    'bear': 'bear_warmap_mask10_anneal_300k',
    'synth_tiger': 'synth_tiger_warmap_mask05_2'
}


scenes = {
    'wolf': 'Walk',
    'beagle_dog': 's1',
    'whiteTiger': 'roaringwalk',
    'fox': 'walk',
    'panda': 'walk',
    'cat': 'walk_final',
    'bear': 'walk'
}

timesteps_dict = {
    'wolf': 0,
    'beagle_dog': 210,
    'whiteTiger': 5,
    'fox': 0,
    'panda': 0,
    'cat': 0,
    'bear': 0,
}

scales_dict = {
    'wolf': '',
    'beagle_dog': 'scale_13.pickle',
    'whiteTiger': 'scale_035.pickle',
    'cat': 'scale.pickle',
    'fox': 'scale_07.pickle',
    'panda': 'scale_045.pickle',
    'bear': 'scale_bear_specific_06.pickle',
}


effective_fur_thickness_cm = {
    
    'panda': {
        "leg_front": 2.5,
        "leg_rear": 2.5,
        "paw_pads": 0.0,
        "front_paws": 1.5,
        "paws": 1.5,
        "belly": 4.0,
        "neck": 4.5,
        "face": 1.5,
        "ears": 2.0,
        "inner_earcanal": 0.2,
        "under_tail": 3.0,
        "eyes": 0.8,
        "tail": 3.5,
        "nosetip": 0.0,
        "body": 3.5
      },

    'fox':  {
        "leg_front": 0.8,
        "leg_rear": 1.2,
        "paw_pads": 0.0,
        "front_paws": 0.3,
        "paws": 0.3,
        "belly": 2.5,
        "neck": 3.5,
        "face": 0.8,
        "ears": 0.5,
        "inner_earcanal": 0.2,
        "under_tail": 4,
        "eyes": 0.3,
        "tail": 6,
        "nosetip": 0.0,
        "body": 2.5
      },

        'cat': {
        "leg_front": 0.3,
        "leg_rear": 0.3,
        "paw_pads": 0.0,
        "front_paws": 0.2,
        "paws": 0.2,
        "belly": 0.5,
        "neck": 0.4,
        "face": 0.2,
        "ears": 0.1,
        "inner_earcanal": 0.05,
        "under_tail": 0.4,
        "eyes": 0,
        "tail": 0.5,
        "nosetip": 0.0,
        "body": 0.4
      },

    'whiteTiger': {
        "leg_front": 1.2,
        "leg_rear": 1.5,
        "paw_pads": 0.0,
        "front_paws": 0.7,
        "paws": 0.7,
        "belly": 4,
        "neck": 3,
        "face": 0.8,
        "ears": 1,
        "inner_earcanal": 0.2,
        "under_tail": 2,
        "eyes": 0.3,
        "tail": 2,
        "nosetip": 0.0,
        "body": 2.5,
        'mane': 5
      },

    'synth_tiger':{
      "leg_front": 0.7,
      "leg_rear": 0.7,
      "paw_pads": 0.0,
      "paws": 0.2,
        "front_paws": 0.2,
      "belly": 1.0,
      "neck": 1.3,
      "face": 0.4,
      "ears": 0.4,
      "inner_ear_canal": 0.4,
      "under_tail": 0.8,
      "eyes": 0.0,
      "tail": 0.8,
      "nose_tip": 0.0,
      "body": 1.0,
      "mane": 1.8
      },

    'beagle_dog': {
        "leg_front": 0.2,
        "leg_rear": 0.2,
        "paw_pads": 0.0,
        "front_paws": 0.1,
        "paws": 0.1,
        "belly": 0.3,
        "neck": 0.4,
        "face": 0.1,
        "ears": 0.5,
        "inner_ear_canal": 0.03,
        "under_tail": 0.3,
        "eyes": 0.05,
        "tail": 0.4,
        "nosetip": 0.0,
        "body": 0.35
    }
}



mapping_gravity_list = {
    'synth_tiger':  {
      "leg_front": [0.0, -0.15, -0.99],   # mostly distal/down, slight toward front
      "leg_rear":  [0.0,  0.00, -1.00],   # distal/down
      "paws":      [0.0, -0.80, -0.60],   # toward toes (front) and down
      "front_paws":[0.0, -0.80, -0.60],   # same as paws; toes point forward
     'paw_pads':       [1, 1, 1],
      "belly":     [0.0,  0.30, -0.95],   # rearward and down
      "neck":      [0.0,  0.85, -0.53],   # from head toward shoulders, slightly down
      "face":      [0.0, -0.75, -0.66],   # toward nose and down (cheeks/chin)
      "ears":      [0.0,  0.00,  1.00],   # outer pinna from base to tip (up)
      "under_tail":[0.0,  0.80, -0.60],   # rearward and down
      "tail":      [0.0,  0.98, -0.20],   # base → tip with slight downward lay
      "body":      [0.0,  0.95, -0.31],   # caudal flow along flanks with slight down
      "mane":      [0.0,  0.20, -0.98],    # cheek/neck ruff hangs down with mild rearward bias
      'inner_earcanal': [1, 1, 1],
      'nosetip':  [1, 1, 1],
      'eyes': [1, 1, 1],
    },

    'beagle_dog': {
        "leg_front": [0.0, -1.0, 0.1],          # Down the leg, slightly forward
        "leg_rear": [0.0, -1.0, -0.1],          # Down the leg, slightly backward (based on haunch shape)
        "paws": [0.0, -1.0, 0.0],               # Straight down (gravity-aligned)
        "front_paws": [0.0, -1.0, 0.0],         # Same as paws — downward
            'paw_pads':       [1, 1, 1],
        "belly": [0.0, -1.0, 0.0],              # Downward from body centerline
        "neck": [0.0, -0.7, 0.7],               # Grows down and forward toward chest
        "face": [0.0, 0.0, 1.0],                # Forward (Z+), radiating from eyes and muzzle center
        "ears": [0.0, -1.0, 0.0],               # Downward along gravity
        "under_tail": [0.0, -0.8, -0.6],        # Grows downward and backward
            'inner_earcanal': [1, 1, 1],
                'nosetip':  [1, 1, 1],
        'eyes': [1, 1, 1],
        "tail": [0.0, -0.1, -1.0],               # From tail base toward tail tip (Z–)
        "body": [0.0, -0.3, 1.0],               # Forward and slightly down (Z+, Y–)
    #     "mane": None                            # No mane observed; omitted
    },

    'whiteTiger':{
        "leg_front":       [0.1, -0.96, -0.19],
        "leg_rear":        [-0.1, -0.94, -0.28],
        'paw_pads':       [1, 1, 1],
        "paws":            [0.28, -0.96, 0.0],      # average toe direction
        "front_paws":      [0.27, -0.93, 0.19],
        "belly":           [0.0, -1.0, 0.0],
        "neck":            [0.0, -0.71, -0.71],
        "face":      [0.0, -0.2, 0.98],
        "ears": [0, 0.85, 0.0],
        'inner_earcanal': [1, 1, 1],
            'nosetip':  [1, 1, 1],
        'eyes': [1, 1, 1],
        "under_tail":      [0.0, -1.0, 0.0],
        "tail":            [0.0, 0.0, -1.0],
        "body":            [0.0, -0.29, -0.96],
        "mane":            [0.0, -0.94, -0.35]
    },

    'cat': {
        'leg_front': [0, -1, 0],
        'leg_rear': [0.0, -1.0, -0.2],
        'paw_pads': [1, 1, 1],
        'paws': [0.0, -1.0, 0.0],
        'front_paws': [0.0, -1.0, 0.0],
        'belly': [0.0, -0.6, -0.8],
        'neck': [0.0, 0.4, 0.9],
        'face': [0.0, 0.2, 1.0],
        'ears': [0.0, 1.0, 0.2],
        'inner_earcanal': [1, 1, 1],
        'under_tail': [0.0, -0.6, -0.9],
        'eyes': [1, 1, 1],
        'tail': [0.0, 1.0, 0.0],
        'nosetip':  [1, 1, 1],
        'body': [0, 0, -1]
        },

    'panda': {
        'leg_front': [0, -1, 0.2],
        'leg_rear': [0, -1, 0.2],
        'paw_pads': [1, 1, 1],
        'paws': [0, -1, 0],
        'front_paws': [0, -1, 0.1],
        'belly': [0.5, -1, 0],
        'neck': [0, -0.2, 1],
        'face': [0, -0.5, 1],
        'ears': [0, 0.3, 1],
        'inner_earcanal': [1, 1, 1],
        'under_tail': [0, -1, -0.3],
        'eyes': [1, 1, 1],
        'tail': [0, -1, -0.5],
        'nosetip':  [1, 1, 1],
        'body': [0, -0.2, -1]
        },
    
    'fox': {
        'leg_front':     [0.0, -0.9,  0.4],
        'leg_rear':      [0.0, -0.85, 0.5],
        'paw_pads': [1, 1, 1],
        'paws':          [0.0, -1.0,  0.0],
        'front_paws':    [0.0, -1.0,  0.0],
        'belly':         [0.0, -1.0,  0.1],
        'neck':          [0.0, -0.5,  0.9],
        'face':          [-0.2, -0.2, 0.95],  # Flip X sign for right cheek
        'ears':          [0.0, -1.0,  0.0],
        'inner_earcanal': [1, 1, 1],
        'under_tail':    [0.0, -1.0,  0.1],
        'tail':          [0.0,  0.0,  1.0],
        'nosetip':  [1, 1, 1],
        'eyes': [1, 1, 1],
        'body':          [0.0, -0.3,  0.95],
    }
}
