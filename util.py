class Config:
    create_directories = True
    download = True
    train = True
    trial = True

class Meta:
    types = ["Creature", "Enchantment", "Artifact", "Instant", "Sorcery",
            "Planeswalker", "Land"
    ]
    test_images = ["IMG_1455.jpeg", "IMG_4945.png", "MessagesImage(339379168).jpeg",
            "MessagesImage(2092370857).png", "MessagesImage(1664742465).jpeg",
            "IMG_1313.JPG", "IMG_1315.JPG", "IMG_1412.JPG", "IMG_1433.JPG",
            "IMG_1438.JPG", "IMG_1440.JPG", "Aeolipile.jpg"
    ]
class Util:
    def cleanName(self, name):
        return name.replace(" ", "_").replace("//", "to")+".jpg"
