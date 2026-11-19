package com.waeil.commence.repo;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.waeil.commence.entity.CartItem;

public interface CartItemRepository extends JpaRepository<CartItem, Long> {
    Optional<CartItem> findByProductNameIgnoreCase(String productName);
    void deleteByProductNameIgnoreCase(String productName);
}