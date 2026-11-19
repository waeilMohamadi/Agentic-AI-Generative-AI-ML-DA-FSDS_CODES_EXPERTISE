package com.waeil.commence.mcp;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.lang.Nullable;
import org.springframework.stereotype.Component;

import com.waeil.commence.service.CartService;

@Component
public class CartTools {

    private final CartService service;

    public CartTools(CartService service) {
        this.service = service;
    }

    @Tool(description = "Add a product to the shopping cart. If it exists, it increases the quantity.")
    public String addToCart(String productName, int quantity) {
        return service.addToCart(productName, quantity);
    }

    @Tool(description = "Remove a product from the cart by product name (e.g., 'iphone').")
    public String removeFromCart(String productName) {
        return service.removeFromCart(productName);
    }

    @Tool(description = "Remove a specified quantity of a product from the cart. If the quantity equals the amount in the cart, remove the product line.")
    public String removeQuantityFromCart(String productName, int quantity) {
        return service.removeQuantityFromCart(productName, quantity);
    }

    @Tool(description = "List all registered products (phones and other items) available in the store, with their prices.")
    public String listProducts() {
        return service.listProducts();
    }

    @Tool(description = "Get the total price of items currently in the cart.")
    public double getCartTotal() {
        return service.getCartTotal();
    }

    @Tool(description = "List items currently in the shopping cart, with quantity, unit price, and line total. Pass a name filter such as 'phone' to return matching products, or leave it empty to list every item.")
    public String listCart(
            @Nullable @ToolParam(description = "Optional product name filter, for example 'phone' or 'iphone'. Leave empty to list the whole cart.", required = false)
            String nameFilter) {
        return service.listCart(nameFilter);
    }
}
