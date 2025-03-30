from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass
class Product(ABC):
    """
    Ancestor class for products
    """
    @property
    @abstractmethod
    def descriptor(self) -> Any:
        pass


@dataclass
class FileProduct(Product, ABC):
    """
    A product that is saved to a file
    """
    store_path: Path

    @property
    def descriptor(self) -> Path:
        return self.store_path

    @classmethod
    @abstractmethod
    def associated_file_extensions(cls) -> List[str]:
        pass


@dataclass
class TextualDataFileProduct(FileProduct):
    """
    A product stores textual data
    """

    @classmethod
    def associated_file_extensions(cls) -> List[str]:
        return ["txt", "csv", "json"]


@dataclass
class FigureFileProduct(FileProduct):
    """
    A product that is saved as a figure
    """
    @classmethod
    def associated_file_extensions(cls) -> List[str]:
        return ["png", "jpg", "jpeg", "pdf", "fig"]


@dataclass
class ModelWeightsFileProduct(FileProduct):
    """
    A product that is saved as model weights
    """
    @classmethod
    def associated_file_extensions(cls) -> List[str]:
        return ["h5"]


class ProductFactory:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProductFactory, cls).__new__(cls)
        return cls._instance

    def create_file_product(self, store_path: Path) -> FileProduct:
        file_extension = store_path.suffix.lstrip('.')
        file_product_classes = FileProduct.__subclasses__()

        for file_product_class in file_product_classes:
            if file_extension in file_product_class.associated_file_extensions():
                return file_product_class(store_path=store_path)
        
        raise ValueError(f"File extension {file_extension} not supported as a file product descriptor")

    def create_from_descriptor(self, descriptor: Any) -> Product:
        if isinstance(descriptor, Path):
            return self.create_file_product(descriptor)
        else:
            raise ValueError(f"Descriptor {descriptor} not supported as a product descriptor")
        

@dataclass
class ExecutionProducts:
    """
    Accumulating container for everyting that is made in the run
    """
    products: List[Product] = field(default_factory=list)

    def _add_product(self, product: Product):
        self.products.append(product)

    def add_product(self, descriptor: Any):
        self._add_product(ProductFactory().create_from_descriptor(descriptor))

    def get_product(self, descriptor: Any) -> Product:
        for product in self.products:
            if product.descriptor == descriptor:
                return product
        raise ValueError(f"Product with descriptor {descriptor} not found")
