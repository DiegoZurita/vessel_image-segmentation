from vessel_images import VesselImages

def main():
    vessel = VesselImages("DRIVE", 50)
    images = vessel.load_images(True)
    masks = vessel.load_masks(True)
    vesselk, not_vessel = vessel.discriminate_pixel_of_mask(masks[0])
    square = vessel.get_square_of_pixel(images[0], not_vessel[74674])

if __name__ == "__main__":
    main()