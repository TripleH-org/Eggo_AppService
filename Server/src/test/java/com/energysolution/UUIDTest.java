package com.energysolution;

import java.util.UUID;

import org.junit.jupiter.api.Test;

public class UUIDTest {
	@Test
	void Test() {
		UUID uuid = UUID.randomUUID();
		System.out.println(uuid);
	}
}
