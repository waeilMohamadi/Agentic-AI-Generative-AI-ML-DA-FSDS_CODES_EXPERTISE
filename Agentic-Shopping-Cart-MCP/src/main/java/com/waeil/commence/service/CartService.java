package com.waeil.commence.service;

import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.waeil.commence.entity.CartItem;
import com.waeil.commence.entity.Product;
import com.waeil.commence.repo.CartItemRepository;
import com.waeil.commence.repo.ProductRepository;

@Service
public class CartService {

    private final CartItemRepository repo;
    private final ProductRepository productRepo;

    // simple in-memory catalog (replace with Product table later)
    private final Map<String, Double> catalog = Map.of(
            "iphone", 999.0,
            "laptop", 1200.0,
            "mouse", 25.0
    );

    public CartService(CartItemRepository repo, ProductRepository productRepo) {
        this.repo = repo;
        this.productRepo = productRepo;
    }

    @Transactional(readOnly = true)
    public String listProducts() {
        List<Product> products = productRepo.findAll();

        if (products.isEmpty()) {
            return "No registered products";
        }

        StringBuilder listing = new StringBuilder();
        for (Product product : products) {
            listing.append(product.getName())
                    .append(" - ")
                    .append(product.getPrice())
                    .append('\n');
        }
        return listing.toString().trim();
    }

    @Transactional
    public String addToCart(String productName, int quantity) {
        if (productName == null || productName.isBlank()) return "Product name is required";
        if (quantity <= 0) return "Quantity must be > 0";

        Double price = catalog.get(productName.toLowerCase());
        if (price == null) return "Product Not Found";

        CartItem item = repo.findByProductNameIgnoreCase(productName)
                .orElseGet(() -> new CartItem(productName, 0, price));

        item.setUnitPrice(price);
        item.setQuantity(item.getQuantity() + quantity);

        repo.save(item);
        return "Added " + quantity + " " + productName;
    }

    @Transactional
    public String removeFromCart(String productName) {
        if (productName == null || productName.isBlank()) return "Product name is required";
        repo.deleteByProductNameIgnoreCase(productName);
        return "Removed " + productName;
    }

    @Transactional
    public String removeQuantityFromCart(String productName, int quantity) {
        if (productName == null || productName.isBlank()) return "Product name is required";
        if (quantity <= 0) return "Quantity must be > 0";

        CartItem item = repo.findByProductNameIgnoreCase(productName.trim()).orElse(null);
        if (item == null) return "Product not in cart";
        if (quantity > item.getQuantity()) return "Cannot remove more than " + item.getQuantity();

        int remaining = item.getQuantity() - quantity;
        if (remaining == 0) {
            repo.delete(item);
        } else {
            item.setQuantity(remaining);
            repo.save(item);
        }
        return "Removed " + quantity + " " + item.getProductName() + "; remaining: " + remaining;
    }

    @Transactional(readOnly = true)
    public double getCartTotal() {
        return repo.findAll().stream()
                .mapToDouble(i -> i.getUnitPrice() * i.getQuantity())
                .sum();
    }

    @Transactional(readOnly = true)
    public String listCart(String nameFilter) {
        String filter = nameFilter == null ? "" : nameFilter.trim().toLowerCase();
        List<CartItem> items = repo.findAll().stream()
                .filter(item -> filter.isEmpty()
                        || item.getProductName().toLowerCase().contains(filter))
                .toList();

        if (items.isEmpty()) {
            return filter.isEmpty() ? "Cart is empty" : "No matching items in the cart";
        }

        StringBuilder listing = new StringBuilder();
        for (CartItem item : items) {
            double lineTotal = item.getUnitPrice() * item.getQuantity();
            listing.append(item.getQuantity())
                    .append(" x ")
                    .append(item.getProductName())
                    .append(" @ ")
                    .append(item.getUnitPrice())
                    .append(" = ")
                    .append(lineTotal)
                    .append('\n');
        }
        return listing.toString().trim();
    }
}
