"""
Handles data loading and preprocessing for the diffusion model.
"""
# imports
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, data_dir:str, batch_size:int, image_size:int, class_cond:bool=False, deterministic:bool=False):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.
    
    NCHW (batch size, channels, height, width): standard image format for PyTorch
        N: number of images in the batch (batch size)
        C: number of channels (3 for RGB images)
        H: height of the image
        W: width of the image
        e.g. (64, 3, 256, 256) for a batch of 64 images, 3 channels, 256x256 pixels
        
    kwargs: additional metadata for each image, e.g. class labels
    Each tensor in kwargs is "batched" - its first dimension matches the batch size
        e.g. If batch_size = 4 (each value in the first tensor corresponds to an image):
        kwargs = {
            "y": tensor([0, 1, 2, 3]),                    # Shape: [4]
            "low_res": tensor([[[...], [...]], ...]),     # Shape: [4, 3, 32, 32]
            "mask": tensor([[...], [...], ...]),          # Shape: [4, 64, 64]
            "text": tensor([101, 102, 103, 104])          # Shape: [4]
        }

    Args:
        data_dir: a dataset directory to load images from.
        batch_size: the batch size of each returned pair.
        image_size: image size.
        class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
        deterministic: if True, yield results in a deterministic order.
    """
    # list all image files in the data directory and its subdirectories recursively
    if not data_dir: # check if data_dir is specified
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir) # returns a list of full paths to all image files found
    
    # check if we want to use conditional classes
    classes = None
    if class_cond: # if we want to use conditional classes
        # Assume classes are the first part of the filename before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files] # extract class names from file paths, e.g. "dog_123.jpg" -> "dog"
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))} # create a dictionary to map class names to unique integer labels, e.g. {"dog": 0, "cat": 1, "bird": 2}
        classes = [sorted_classes[x] for x in class_names] # convert class names to their corresponding integer labels, e.g. ["dog", "cat", "bird"] -> [0, 1, 2]
    
    # create ImageDataset object
    dataset = ImageDataset(
        image_size, # image size
        all_files, # list of image file paths
        classes=classes, # class labels
        shard=MPI.COMM_WORLD.Get_rank(), # rank of the current process (low level programming)
        num_shards=MPI.COMM_WORLD.Get_size(), # total number of processes (low level programming)
    )
    
    # create data loader, either deterministic or shuffled
    if deterministic: # if we want to load data in a deterministic order, not shuffled
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else: # if we want to load data in a random order
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True: # infinite loop to yield batches from the loader, continuous data flow
        yield from loader


def _list_image_files_recursively(data_dir:str) -> list[str]:
    """
    Gathers all image files in the given directory and its subdirectories.
    Args:
        data_dir: the directory to search for image files.
    Returns:
        A list of full paths to all image files found.
    """
    # get paths to all image files in the directory
    results = []
    for entry in sorted(bf.listdir(data_dir)): # iterate over all entries in the directory, sorted alphabetically
        full_path = bf.join(data_dir, entry) # construct full path to the entry
        ext = entry.split(".")[-1] # extract the file extension, e.g. ".jpg"
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]: # check if the entry is an image file
            results.append(full_path)
        elif bf.isdir(full_path): # if the entry is a directory, recursively search for image files
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    """
    Custom dataset class for loading and preprocessing images.
    """
    def __init__(self, resolution:int, image_paths:list[str], classes:list[int]=None, shard:int=0, num_shards:int=1):
        super().__init__() # inherit from the base class Dataset
        self.resolution = resolution # image size
        self.local_images = image_paths[shard:][::num_shards] # distribute images across shards
        self.local_classes = None if classes is None else classes[shard:][::num_shards] # distribute class labels across shards

    def __len__(self):
        return len(self.local_images) # return the number of images in the current shard

    def __getitem__(self, idx:int) -> tuple[np.ndarray, dict]:
        '''
        Loads and preprocesses an image from the dataset.
        Args:
            idx: the index of the image to load.
        Returns:
            A tuple containing the preprocessed image and its metadata
        '''
        # opens and loads the image from the file
        path = self.local_images[idx] # get the path to the image at the given index
        with bf.BlobFile(path, "rb") as f: # open the image file in binary mode
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        # downsample the image to the desired resolution
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        # final resizing of the image to the desired resolution
        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        # convert the image to a numpy array and crop it to the desired resolution
        arr = np.array(pil_image.convert("RGB")) # convert the image to a numpy array
        crop_y = (arr.shape[0] - self.resolution) // 2 # calculate the y-coordinate of the crop
        crop_x = (arr.shape[1] - self.resolution) // 2 # calculate the x-coordinate of the crop
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution] # crop the image to the desired resolution
        arr = arr.astype(np.float32) / 127.5 - 1 # normalize the image to the range [-1, 1]

        # create a dictionary to store the image and its metadata
        out_dict = {}
        if self.local_classes is not None: # if class labels are available
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64) # add the class label to the dictionary
    
        return np.transpose(arr, [2, 0, 1]), out_dict # return the image and its metadata, e.g. arr.shape = (3, 256, 256) and out_dict = {"y": 0}
