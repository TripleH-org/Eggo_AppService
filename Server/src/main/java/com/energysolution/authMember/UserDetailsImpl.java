package com.energysolution.authMember;

import java.util.ArrayList;
import java.util.Collection;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.authority.SimpleGrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import com.energysolution.dto.UserDTO;
import com.energysolution.dto.UserRoleDTO;
import com.energysolution.model.UserRole;

import lombok.AllArgsConstructor;


@AllArgsConstructor
public class UserDetailsImpl implements UserDetails {
	//UserDetails : 사용자 정보를 담는 인터페이스
	
	private UserRoleDTO userRole;
	private UserDTO user;

	@Override
	public Collection<? extends GrantedAuthority> getAuthorities() {
		UserRole role = userRole.getRole();
		SimpleGrantedAuthority authority = new SimpleGrantedAuthority(role.getRoleType());
		Collection<GrantedAuthority> authorities = new ArrayList<>();
		authorities.add(authority);
		
		return authorities;
	}

	@Override
	public String getPassword() {
		return user.getPassword();
	}

	@Override
	public String getUsername() {
		return user.getUserId();
	}

	@Override
	public boolean isAccountNonExpired() {
		return true;
	}

	@Override
	public boolean isAccountNonLocked() {
		return true;
	}

	@Override
	public boolean isCredentialsNonExpired() {
		return true;
	}

	@Override
	public boolean isEnabled() {
		return true;
	}

}
