package com.energysolution.service;

import java.io.UnsupportedEncodingException;
import java.security.SecureRandom;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.mail.MessagingException;
import javax.mail.internet.MimeMessage;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.mail.javamail.MimeMessageHelper;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.energysolution.authMember.UserDetailsImpl;
import com.energysolution.dto.UserDTO;
import com.energysolution.dto.UserRoleDTO;
import com.energysolution.mapper.UserMapper;

@Service
public class UserServiceImpl implements UserService {

	@Autowired
	private UserMapper userMapper;
	
    @Autowired
    private JavaMailSender javaMailSender;
    
    @Autowired
    private PasswordEncoder passwordEncoder;
    
    @Autowired
	private AuthenticationManager authManager;
    
	//회원가입
	@Override
	public String registerUser(UserDTO userDTO) {
		userDTO.setPassword(passwordEncoder.encode(userDTO.getPassword()));
		userMapper.insertUser(userDTO);
		userMapper.insertRole(new UserRoleDTO(userDTO.getUserId()));
		
		if(checkUser(userDTO.getUserId())==1)
			return "success";
		return "fail";
	}

	//로그인 하기
	@Override
	public UserDTO loginUser(String UserId, String Password) {	
		Authentication auth = authManager.authenticate(
				new UsernamePasswordAuthenticationToken(UserId, Password)
				);
		SecurityContextHolder.getContext().setAuthentication(auth);
		UserDetailsImpl principal = (UserDetailsImpl) auth.getPrincipal();
		//해당 아이디의 비밀번호가 사용자가 입력한 비밀번호와 일치하는지 확인
		//암호화된 비밀번호
		if(checkUser(UserId)==0)
			return new UserDTO();
		if(passwordEncoder.matches(Password, getUserPassword(UserId))) {
			UserDTO userDTO = userMapper.getUser(principal.getUsername());
			return userDTO;
		}
		return new UserDTO();
	}	

	//비밀번호 변경
	@Override
	public String updateUser(HashMap<String, String> updateMap) {
		String encodedPW = passwordEncoder.encode(updateMap.get("newPW"));
		updateMap.remove("newPW");
		updateMap.put("newPW", encodedPW);
		//새 비밀번호로 변경
		userMapper.updateUser(updateMap);
		
		//아이디로 비밀번호 조회
		String getPassword = userMapper.getUserPassword(updateMap.get("UserId"));
		
		//새 비밀번호로 바뀌었는지 확인
		if(updateMap.get("newPW").equals(getPassword))
			return "success";
		return "fail";
	}
	
	//아이디 찾기
	@Override
	public List<String> FindUserId(String Email) {
		return userMapper.FindUserId(Email);
	}
	
	//비밀번호 찾기
	@Override
	public String FindUserPW(HashMap<String, String> findUserPWMap) throws MessagingException, UnsupportedEncodingException {
		//아이디 있는지 확인
		int count = userMapper.checkUserByIdEmail(findUserPWMap);
		System.out.println(count);
		
		if(count==0)
			return "fail";
		
		UserDTO userDTO = userMapper.FindUserPW(findUserPWMap);
		
		//임시 비밀번호 발급
		String tempPW = getRandomPassword(10);
		String encodedPW = passwordEncoder.encode(tempPW);
		
		//이메일 전송
		String to = userDTO.getEmail();
        String from = "gmlwn@test.com";
        String subject = "temporary password";
        
        StringBuilder body = new StringBuilder();
        body.append("<html> <body><h1>임시 비밀번호는 다음과 같습니다. </h1> <div>");
        body.append(tempPW);
        body.append("</div> </body></html>");

        MimeMessage message = javaMailSender.createMimeMessage();
        MimeMessageHelper mimeMessageHelper = new MimeMessageHelper(message, true, "UTF-8");

        mimeMessageHelper.setFrom(from,"EnergySolution_TempPW_manager");
        mimeMessageHelper.setTo(to);
        mimeMessageHelper.setSubject(subject);
        mimeMessageHelper.setText(body.toString(), true);

        javaMailSender.send(message);
		
        // DB 변경
		HashMap<String, String> updateMap = new HashMap<String, String>();
		updateMap.put("UserId", userDTO.getUserId());
		updateMap.put("newPW", encodedPW);
		String res = updateUser(updateMap);
		
		return res;
	}
	
	//임시 비밀번호 발급
	public String getRandomPassword(int size) {
		char[] charSet = new char[] { 
				'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
				'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
				'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
				'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
				'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
				'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
				'y', 'z', '!', '@', '#', '$', '%', '^', '&' }; 
		StringBuffer sb = new StringBuffer(); 
		SecureRandom sr = new SecureRandom(); 
		sr.setSeed(new Date().getTime()); 
		int idx = 0; 
		int len = charSet.length;
		for (int i=0; i<size; i++) {
			// idx = (int) (len * Math.random()); 
			idx = sr.nextInt(len); 
			// 강력한 난수를 발생시키기 위해 SecureRandom을 사용 
			sb.append(charSet[idx]); 
			} 
		return sb.toString(); 
	}


	@Override
	public String deleteUser(String UserId,String Password) {
		if(!Password.equals(getUserPassword(UserId)))
			return "fail";
		
		userMapper.deleteUser(UserId);
		
		//UserId에 해당하는 User 못찾았다면 success
		if(checkUser(UserId)==0)
			return "success";
		return "fail";
	}
	
	
	// UserId로 User있는지 확인
	@Override
	public int checkUser(String UserId) {
		return userMapper.checkUser(UserId);
	}
	
	@Override
	public int checkUserByIdEmail(HashMap<String, String> UserMap) {
		return userMapper.checkUserByIdEmail(UserMap);
	}
	
	// DB에서 암호화된 비밀번호 가져오기
	@Override
	public String getUserPassword(String UserId) {
		return userMapper.getUserPassword(UserId);
	}

	@Override
	public List<Map<String, Object>> getReqData() throws Exception {
		return userMapper.getReqData();
	}
}
