package com.waeil.commence.service;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.*;

import java.util.Optional;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import com.waeil.commence.entity.CartItem;
import com.waeil.commence.repo.CartItemRepository;
import com.waeil.commence.repo.ProductRepository;

@ExtendWith(MockitoExtension.class)
class CartServiceTest {
    @Mock CartItemRepository cartRepository;
    @Mock ProductRepository productRepository;
    @InjectMocks CartService service;

    @Test
    void removesSomeAndKeepsLine() {
        CartItem item = new CartItem("iPhone", 100, 999.0);
        when(cartRepository.findByProductNameIgnoreCase("iPhone")).thenReturn(Optional.of(item));
        assertEquals("Removed 20 iPhone; remaining: 80", service.removeQuantityFromCart("iPhone", 20));
        assertEquals(80, item.getQuantity());
        verify(cartRepository).save(item);
        verify(cartRepository, never()).delete(item);
    }

    @Test
    void removesLineWhenQuantityMatches() {
        CartItem item = new CartItem("iPhone", 20, 999.0);
        when(cartRepository.findByProductNameIgnoreCase("iPhone")).thenReturn(Optional.of(item));
        assertEquals("Removed 20 iPhone; remaining: 0", service.removeQuantityFromCart("iPhone", 20));
        verify(cartRepository).delete(item);
    }

    @Test
    void rejectsOverRemovalWithoutMutation() {
        CartItem item = new CartItem("iPhone", 10, 999.0);
        when(cartRepository.findByProductNameIgnoreCase("iPhone")).thenReturn(Optional.of(item));
        assertEquals("Cannot remove more than 10", service.removeQuantityFromCart("iPhone", 20));
        verify(cartRepository, never()).save(item);
        verify(cartRepository, never()).delete(item);
    }
}
