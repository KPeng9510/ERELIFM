import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
from torch.utils.data import Dataset
from torchvision.datasets.folder import has_file_allowed_extension, find_classes, make_dataset, default_loader
import torch
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def make_dataset_new(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    intervals = []
    num_instances = 0
    available_classes = set()
    for target_class in class_to_idx.keys():
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

            intervals.append((num_instances, num_instances+len(fnames)-1))
            num_instances += len(fnames)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances#, intervals




class SingleClassData_2(Dataset):
    def __init__(self, root_dir, domain, classes, domain_label=-1, classes_label=-1, transform=None, loader=default_loader):          
        
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  

        super().__init__()
        self.root_dir = root_dir
        self.domain_name = domain
        self.class_name = classes
        self.domain_label = domain_label
        self.classes_label = classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.length = 0
        
        self.load_dataset()

    def load_dataset(self):
        class_to_idx = {self.class_name: self.classes_label}      
        path = os.path.join(self.root_dir, self.domain_name)

        if not os.path.isdir(path):
            raise ValueError("Domain \"{}\" does not exit.".format(self.domain_name))  

        self.samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        self.length = len(self.samples)

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.domain_label




class SingleClassData(Dataset):
    def __init__(self, root_dir, domain,  classes, train, nsr,all_classes=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house'], asymetric=False, domain_label=-1, classes_label=-1, transform=None, loader=default_loader):          
        
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  

        super().__init__()
        self.root_dir = root_dir
        self.domain_name = domain
        self.class_name = classes
        self.all_classes = all_classes
        self.domain_label = domain_label
        self.classes_label = classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.length = 0
        

        if train == True:
            self.load_dataset()

            import copy
            self.original_samples = copy.deepcopy(self.samples)
            if asymetric == False:
                if nsr == 0.8:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_symetric_snr_0_8.pkl"
                if nsr == 0.5:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_symetric_snr_0_5.pkl"
                if nsr == 0.2:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_symetric_snr_0_2.pkl"
            else:
                file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_asymetric_snr_0_5.pkl"
            import pickle as pkl
            f = open(file_name, 'rb')
            labels = pkl.load(f)
            f.close()
            #print(labels)
            self.samples_crr = []
            #k = 0
            for class_label, dataset in enumerate(self.all_data):

                for i, (data, l) in enumerate(dataset):
                    try:
                        if labels[domain][class_label][i] == classes_label:
                            self.samples_crr.append([data, labels[domain][class_label][i]])
                    except:
                        print('error')
            self.samples = self.samples_crr
            self.length = len(self.samples)

        else:
            self.load_dataset2()
    def load_dataset(self):
        self.all_data = []
        for name in self.all_classes:
            #print(name)
            class_to_idx = {name: self.all_classes.index(name)}      
            path = os.path.join(self.root_dir, self.domain_name)

            if not os.path.isdir(path):
                raise ValueError("Domain \"{}\" does not exit.".format(self.domain_name))  
            
            self.all_data.append(make_dataset_new(path, class_to_idx, IMG_EXTENSIONS))

    def load_dataset2(self):
        class_to_idx = {self.class_name: self.classes_label}      
        path = os.path.join(self.root_dir, self.domain_name)

        if not os.path.isdir(path):
            raise ValueError("Domain \"{}\" does not exit.".format(self.domain_name))  

        self.samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        self.length = len(self.samples)
    def index_reset(self, indexes):
        self.old_samples = self.samples
        indexes = list(indexes)
        self.samples = [self.samples[ind] for ind in indexes]
        self.length = len(self.samples)
        #self.original_samples = [self.original_samples[ind] for ind in indexes]
    def index_reset_all(self):
        self.samples = self.old_samples
        self.length = len(self.samples)       
    def set_transform(self, transform):
        self.transform = transform
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        path, label = self.samples[index]
        #_, label_orin = self.original_samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, label


class SingleDomainData(Dataset):
    def __init__(self, root_dir, domain, classes, train, nsr, asymetric=False, domain_label=-1, get_classes_label=True, class_to_idx=None, transform=None, loader=default_loader):          
        
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  

        super().__init__()
        self.root_dir = root_dir
        self.domain_name = domain
        self.classes = classes
        self.class_to_idx = class_to_idx 
        self.domain_label = domain_label
        self.get_classes_label = get_classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.length = 0
        self.load_dataset()
        if train == True:
            if asymetric == False:
                if nsr == 0.8:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_symetric_snr_0_8.pkl"
                if nsr == 0.5:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_symetric_snr_0_5.pkl"
                if nsr == 0.2:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_symetric_snr_0_2.pkl"
            else:
                file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/PACS_unseen_person_asymetric_snr_0_5.pkl"
            import pickle as pkl
            f = open(file_name, 'rb')
            labels = pkl.load(f)
            f.close()
            
            self.samples_crr = []
            for i, (data, l) in enumerate(self.samples):
                self.samples_crr.append([data, labels[i]])
            self.samples = self.samples_crr

    def load_dataset(self):
        if self.get_classes_label == False:
            class_to_idx = {self.classes[i]: -1 for i in range(len(self.classes))}           
        elif self.class_to_idx is None:
            class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            class_to_idx = self.class_to_idx
            
        path = os.path.join(self.root_dir, self.domain_name)

        if not os.path.isdir(path):
            raise ValueError("Domain \"{}\" does not exit.".format(self.domain_name))  

        self.samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        self.length = len(self.samples)

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.domain_label, index

class MultiDomainData(Dataset):
    def __init__(self, root_dir, domain, classes,nsr,asymetric=False,all_classes=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house'],transform2=None, domain_class_dict=None, train=False, get_domain_label=True, get_classes_label=True, transform=None, loader=default_loader, simclr_transform=None):          
        import os
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  
        super().__init__()
        self.root_dir = root_dir
        self.domain = sorted(domain)
        self.classes = classes
        print(self.classes)
        #self.all_domains =  ["clipart",  "infograph",  "painting",  "quickdraw",  "real",  "sketch"]
        self.transform2 = transform2
        self.all_classes = self.classes
        self.domain_class_dict = domain_class_dict
        self.get_domain_label = get_domain_label
        self.get_classes_label = get_classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.train = train
        self.domain_label = []
        self.load_dataset()
        self.load_dataset_all()
        self.simclr_transform = simclr_transform
        if train == True:
            if asymetric == False:
                if nsr == 0.8:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/domainnet_symetric_snr_0_8.pkl"
                if nsr == 0.5:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/domainnet_symetric_snr_0_5.pkl"
                if nsr == 0.2:
                    file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/domainnet_symetric_snr_0_2.pkl"
            else:
                file_name = "/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/domainnet_asymetric_snr_0_5.pkl"

            import pickle as pkl

            f = open(file_name, 'rb')
            labels = pkl.load(f)
            f.close()


            self.samples_origin = []#self.samples
            #print(labels)
            self.samples_crr = []
            #k = 0
            #print(labels.keys())
            self.domain_label = []
            #os.exit()
            for class_label, domain_dataset in enumerate(self.all_data):
                #print(len(dataset))
                #print(len(labels[d][class_label]))
                for k, dataset in enumerate(domain_dataset):
                    d = self.domain[k]
                    for i, (data, l) in enumerate(dataset):
                        #try:
                        #if labels[d][class_label][i] == classes_label:
                        
                        #print(len(dataset))
                        #print(len(labels[d][class_label]))

                        self.samples_crr.append([data, labels[d][class_label][i]])
                        self.samples_origin.append([data, l])
                        self.domain_label.append(self.domain.index(d))
                        #except:
                        #    print('error')
                self.samples = self.samples_crr
                self.length = len(self.samples)
        else:
            self.load_dataset2()
        self.labels = torch.Tensor([i for _,i in self.samples])
        self.domain_label = torch.Tensor(self.domain_label)

    def load_dataset(self):
        if self.get_classes_label:
            class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            class_to_idx = {self.classes[i]: -1 for i in range(len(self.classes))}
        
        for i, domain_name in enumerate(self.domain): 
            path = os.path.join(self.root_dir, domain_name)

            if not os.path.isdir(path):
                raise ValueError("Domain \"{}\" does not exit.".format(domain_name))  

            if self.domain_class_dict is None:
                sub_class_to_idx = class_to_idx
            else:
                sub_class_to_idx = {the_class: class_to_idx[the_class] for the_class in self.domain_class_dict[domain_name]}
            
            samples = make_dataset(path, sub_class_to_idx, IMG_EXTENSIONS)
            self.samples.extend(samples)
            if self.get_domain_label: 
                domain_label = [i] * len(samples)
                self.domain_label.extend(domain_label)

    def load_dataset_all(self):
        self.all_data = []
        for name in self.classes:
            #print(name)
            class_to_idx = {name: self.classes.index(name)}
            domain_datasets = []
            for domain_name in self.domain:  
                path = os.path.join(self.root_dir, domain_name)
                if not os.path.isdir(path):
                    raise ValueError("Domain \"{}\" does not exit.".format(domain_name))  
                domain_datasets.append(make_dataset_new(path, class_to_idx, IMG_EXTENSIONS))
            self.all_data.append(domain_datasets)

    def load_dataset2(self):
        self.all_data = []
        for name in self.all_classes:
            #print(name)
            class_to_idx = {name: self.all_classes.index(name)}      
            for domain_name in self.domain:  
                path = os.path.join(self.root_dir, domain_name)

                if not os.path.isdir(path):
                    raise ValueError("Domain \"{}\" does not exit.".format(domain_name))  
                
                self.all_data.append(make_dataset_new(path, class_to_idx, IMG_EXTENSIONS))
    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        #print(index)
        path, label = self.samples[index]
        _, label_orin = self.samples_origin[index]
        img = self.loader(path)
        #orin_img =img
        orin_img = self.transform(img)
        if self.transform2 == None:
            self.transform2 = self.transform
        img2 = self.transform2(img)
        if self.train == True:
            if self.simclr_transform is not None:
                img = self.simclr_transform(img)

        target = self.domain_label[index] if self.get_domain_label else -1
        if self.train == True:
            return orin_img,img2, label, label_orin , target, index , [path]  #orin_img,img2, label, index
        else:
            return orin_img,img2, label, index#label_orin, target, torch.Tensor(index)


class MultiDomainData_with_no_noise(Dataset):
    def __init__(self, root_dir, domain, classes,nsr,asymetric=False,all_classes=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house'], domain_class_dict=None, train=False, get_domain_label=True, get_classes_label=True, transform=None, loader=default_loader, simclr_transform=None):          
        import os
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  

        super().__init__()
        self.root_dir = root_dir
        self.domain = sorted(domain)
        self.classes = classes
        self.all_classes = all_classes
        self.domain_class_dict = domain_class_dict
        self.get_domain_label = get_domain_label
        self.get_classes_label = get_classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.domain_label = []
        self.load_dataset()
        self.load_dataset_all()
        self.simclr_transform = simclr_transform
        self.load_dataset2()
        self.train = train

    def load_dataset(self):
        if self.get_classes_label:
            class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            class_to_idx = {self.classes[i]: -1 for i in range(len(self.classes))}
        
        for i, domain_name in enumerate(self.domain): 
            path = os.path.join(self.root_dir, domain_name)

            if not os.path.isdir(path):
                raise ValueError("Domain \"{}\" does not exit.".format(domain_name))  

            if self.domain_class_dict is None:
                sub_class_to_idx = class_to_idx
            else:
                sub_class_to_idx = {the_class: class_to_idx[the_class] for the_class in self.domain_class_dict[domain_name]}
            
            samples = make_dataset(path, sub_class_to_idx, IMG_EXTENSIONS)
            self.samples.extend(samples)
            if self.get_domain_label: 
                domain_label = [i] * len(samples)
                self.domain_label.extend(domain_label)

    def load_dataset_all(self):
        self.all_data = []
        for name in self.all_classes:
            #print(name)
            class_to_idx = {name: self.all_classes.index(name)}
            domain_datasets = []
            for domain_name in self.domain:  
                path = os.path.join(self.root_dir, domain_name)

                if not os.path.isdir(path):
                    raise ValueError("Domain \"{}\" does not exit.".format(domain_name))  
                
                domain_datasets.append(make_dataset_new(path, class_to_idx, IMG_EXTENSIONS))
            self.all_data.append(domain_datasets)
    def load_dataset2(self):
        self.all_data = []
        for name in self.all_classes:
            #print(name)
            class_to_idx = {name: self.all_classes.index(name)}      
            for domain_name in self.domain:  
                path = os.path.join(self.root_dir, domain_name)

                if not os.path.isdir(path):
                    raise ValueError("Domain \"{}\" does not exit.".format(domain_name))  
                
                self.all_data.append(make_dataset_new(path, class_to_idx, IMG_EXTENSIONS))
    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        #_, label_orin = self.samples_origin[index]
        img = self.loader(path)
        orin_img = img
        orin_img = self.transform(orin_img)
        if self.train==True:
            if self.simclr_transform is not None:
                img = self.simclr_transform(img)

        target = self.domain_label[index] if self.get_domain_label else -1
        candidate_list = [0,1,2,3,4,5]
        candidate_list.remove(label)
        index = torch.randint(0, 5, (1,))[0]
        return torch.Tensor(orin_img), img, label,candidate_list[index], target, torch.Tensor(index) #



if __name__ == '__main__':
    from random import sample
    import torch
    import numpy as np
    from text2vec import SentenceModel
    domain_list = ["clipart",  "infograph",  "painting",  "quickdraw",  "real",  "sketch"]
    class_list_all = [
    "aircraft_carrier", "belt", "camel", "cow", "face", "hamburger", "knee", "motorbike", "peas", "rifle", "snorkel", "sweater", "trombone",
    "airplane", "bench", "camera", "crab", "fan", "hammer", "knife", "mountain", "pencil", "river", "snowflake", "swing_set", "truck",
    "alarm_clock", "bicycle", "camouflage", "crayon", "feather", "hand", "ladder", "mouse", "penguin", "roller_coaster", "snowman", "sword", "trumpet",
    "ambulance", "binoculars", "campfire", "crocodile", "fence", "harp", "lantern", "moustache", "piano", "rollerskates", "soccer_ball", "syringe", 
    "angel", "bird", "candle", "crown", "finger", "hat", "laptop", "mouth", "pickup_truck", "sailboat", "sock", "table", "umbrella",
    "animal_migration", "birthday_cake", "cannon", "cruise_ship", "fire_hydrant", "headphones", "leaf", "mug", "picture_frame", "sandwich", "speedboat", "teapot", "underwear",
    "ant", "blackberry", "canoe", "cup", "fireplace", "hedgehog", "leg", "mushroom", "pig", "saw", "spider", "teddy-bear", "van",
    "anvil", "blueberry", "car", "diamond", "firetruck", "helicopter", "light_bulb", "nail", "pillow", "saxophone", "spoon", "telephone", "vase",
    "apple", "book", "carrot", "dishwasher", "fish", "helmet", "lighter", "necklace", "pineapple", "school_bus", "spreadsheet", "television", "violin",
    "arm", "boomerang", "castle", "diving_board", "flamingo", "hexagon", "lighthouse", "nose", "pizza", "scissors", "square", "tennis_racquet", "washing_machine",
    "asparagus", "bottlecap", "cat", "dog", "flashlight", "hockey_puck", "lightning", "ocean", "pliers", "scorpion", "squiggle", "tent", "watermelon",
    "axe", "bowtie", "ceiling_fan", "dolphin", "flip_flops", "hockey_stick", "line", "octagon", "police_car", "screwdriver", "squirrel", "The_Eiffel_Tower", "waterslide",
    "backpack", "bracelet", "cello", "donut", "floor_lamp", "horse", "lion", "octopus", "pond", "sea_turtle", "stairs", "The_Great_Wall_of_China", "whale",
    "banana", "brain", "cell_phone", "door", "flower", "hospital", "lipstick", "onion", "pool", "see_saw", "star", "The_Mona_Lisa", "wheel",
    "bandage", "bread", "chair", "dragon", "flying_saucer", "hot_air_balloon", "lobster", "oven", "popsicle", "shark", "steak", "tiger", "windmill",
    "barn", "bridge", "chandelier", "dresser", "foot", "hot_dog", "lollipop", "owl", "postcard", "sheep", "stereo", "toaster", "wine_bottle",
    "baseball", "broccoli", "church", "drill", "fork", "hot_tub", "mailbox", "paintbrush", "potato", "shoe", "stethoscope", "toe", "wine_glass",
    "baseball_bat", "broom", "circle", "drums", "frog", "hourglass", "map", "paint_can", "power_outlet", "shorts", "stitches", "toilet", "wristwatch",
    "basket", "bucket", "clarinet", "duck", "frying_pan", "house", "marker", "palm_tree", "purse", "shovel", "stop_sign", "tooth", "yoga",
    "basketball", "bulldozer", "clock", "dumbbell", "garden", "house_plant", "matches", "panda", "rabbit", "sink", "stove", "toothbrush", "zebra",
    "bat", "bus", "cloud", "ear", "garden_hose", "hurricane", "megaphone", "pants", "raccoon", "skateboard", "strawberry", "toothpaste", "zigzag",
    "bathtub", "bush", "coffee_cup", "elbow", "giraffe", "ice_cream", "mermaid", "paper_clip", "radio", "skull", "streetlight", "tornado",
    "beach", "butterfly", "compass", "elephant", "goatee", "jacket", "microphone", "parachute", "rain", "skyscraper", "string_bean", "tractor",
    "bear", "cactus", "computer", "envelope", "golf_club", "jail", "microwave", "parrot", "rainbow", "sleeping_bag", "submarine", "traffic_light",
    "beard", "cake", "cookie", "eraser", "grapes", "kangaroo", "monkey", "passport", "rake", "smiley_face", "suitcase", "train",
    "bed", "calculator", "cooler", "eye", "grass", "key", "moon", "peanut", "remote_control", "snail", "sun", "tree",
    "bee", "calendar", "couch", "eyeglasses", "guitar", "keyboard", "mosquito", "pear", "rhinoceros", "snake", "swan", "triangle", "t-shirt",
    ]
    class_list = sorted(class_list_all)

    class_list = class_list_all[:182]
    unseen_class_list = class_list_all[182:]
    all_labels = {}
    #"raccoon" # "squirrel"
    pre_set_snr = 0.5
    all_labels = {}
    ###########################
    #### Code for symetric label generation
    #############################
    for domain in domain_list:
        print(domain)
        single_domain_labels = []
        for c, category in enumerate(class_list):
            print(category)
            dataset = SingleClassData_2(root_dir='/hkfs/work/workspace/scratch/fy2374-got/domainnet/DMG/data/DomainNet/train', domain=domain, classes=category,  classes_label=c)
            symetric_labels = []
            label_noise_candidate = []
            for k in class_list:
                if k != category:
                    label_noise_candidate.append(class_list.index(k))
            data_len = len(dataset)
            index_list = list(range(0, data_len))
            noisy_list = sample(index_list, int(pre_set_snr*data_len))
            for index, (_, label, _) in enumerate(dataset):
                if index in noisy_list:
                    symetric_labels.append(sample(label_noise_candidate,1)[0])
                else:
                    symetric_labels.append(label)
            single_domain_labels.append(symetric_labels)
        all_labels[domain] = single_domain_labels
    import pickle as pkl
    f = open('/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/domainnet_symetric_snr_0_5.pkl', 'wb')
    pkl.dump(file=f, obj=all_labels)
    f.close()


    ###########################
    #### Code for asymetric label generation
    #############################
    #sbert_model = SentenceModel("shibing624/text2vec-base-multilingual")
    '''embeddings = class_list#sbert_model.encode(class_list)
    pre_set_snr = 0.5
    #print(embeddings.shape)
    from text2vec import Similarity
    sim_model = Similarity()
    noisy_set = []
    for i in range(len(embeddings)):
        score_list = []
        for j in range(len(embeddings)):
            score = sim_model.get_score(embeddings[i], embeddings[j])
            score_list.append(score)
        score_list = np.array(score_list)
        ind = np.argmin(score_list)
        noisy_set.append(class_list[ind])

    for domain in domain_list:
        single_domain_labels = []
        print(domain)
        for c, category in enumerate(class_list):
            print(category)
            dataset = SingleClassData_2(root_dir='/hkfs/work/workspace/scratch/fy2374-got/domainnet/DMG/data/DomainNet/train', domain=domain, classes=category,  classes_label=c)
            symetric_labels = []
            label_noise_candidate = noisy_set[c]

            data_len = len(dataset)
            index_list = list(range(0, data_len))
            noisy_list = sample(index_list, int(pre_set_snr*data_len))
            for index, (_, label, _) in enumerate(dataset):
                #p = torch.rand(1)[0]
                #print(p)
                if index in noisy_list:
                    
                    symetric_labels.append(class_list.index(label_noise_candidate))
                else:
                    symetric_labels.append(label)
            single_domain_labels.append(symetric_labels)
        all_labels[domain] = single_domain_labels
    import pickle as pkl
    f = open('/hkfs/work/workspace/scratch/fy2374-got/workspace/ijcai_folders/Neurips/benchmarks_noisy_label_osdg/domainnet_asymetric_snr_0_5.pkl', 'wb')
    pkl.dump(file=f, obj=all_labels)
    f.close()'''