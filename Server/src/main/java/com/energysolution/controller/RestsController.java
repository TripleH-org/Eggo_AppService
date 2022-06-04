package com.energysolution.controller;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.List;

import javax.mail.MessagingException;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import com.energysolution.dto.TotalBillDTO;
import com.energysolution.dto.UserDTO;
import com.energysolution.service.BillServiceImpl;
import com.energysolution.service.UserServiceImpl;

@RestController
public class RestsController {
	
	private String result;
	private List<String> userIdList;

	@Autowired
	UserServiceImpl userService;
		
	@Autowired
	BillServiceImpl billService;
	
	private JSONObject jsonObject;

	// 비밀번호 바꾸기
	@PostMapping("/main/changePW")
	public JSONObject updateUser(@RequestBody UserDTO userDTO) {
		System.out.println("비밀번호 변경 시도 : " + userDTO.getUserId());
		HashMap<String, String> updateMap = new HashMap<String, String>();
		updateMap.put("UserId", userDTO.getUserId());
		updateMap.put("newPW", userDTO.getPassword());

		result = userService.updateUser(updateMap);

		JSONObject resultJSON = new JSONObject();

		if (result == "fail") {
			resultJSON.put("message", "비밀번호 변경 실패");
		} else {
			resultJSON.put("message", "비밀번호 변경 완료");
		}

		return resultJSON;
	}

	// ID 찾기
	@GetMapping("/main/findUserId")
	public JSONObject FindUserId(@RequestParam("email") String Email) {

		userIdList = userService.FindUserId(Email);

		JSONObject resultJSON = new JSONObject();
		resultJSON.put("message", "아이디 찾기 성공");

		JSONObject data = new JSONObject();
		data.put("count", userIdList.size());

		JSONArray Idarr = new JSONArray();
		for (String userId : userIdList) {
			JSONObject Id = new JSONObject();
			Id.put("UserId", userId);
			Idarr.add(Id);
		}

		data.put("UserIdList", Idarr);

		resultJSON.put("data", data);

		return resultJSON;
	}

	// 비밀번호 찾기
	@GetMapping("/main/findUserPW")
	public JSONObject FindUserPW(@RequestParam("userId") String UserId, @RequestParam("email") String Email)
			throws UnsupportedEncodingException, MessagingException {

		HashMap<String, String> findUserPWMap = new HashMap<String, String>();
		findUserPWMap.put("UserId", UserId);
		findUserPWMap.put("Email", Email);
		result = userService.FindUserPW(findUserPWMap);

		if (result == "fail")
			return null;

		/* 이메일로 전송하기 */

		JSONObject resultJSON = new JSONObject();
		resultJSON.put("Message", "이메일로 해당 아이디의 임시 비밀번호가 전송되었습니다.");

		JSONObject data = new JSONObject();
		data.put("UserId", UserId);

		resultJSON.put("data", data);

		return resultJSON;
	}


	// 탈퇴 기능 나중에 논의
	@RequestMapping("deleteUser")
	public @ResponseBody String deleteUser() {

		// result =
		// userService.deleteUser((String)obj.get("UserId"),(String)obj.get("Password"));

		// Payment - Bill 테이블과 얽혀있어서 더 구현해야함!

		if (result == "fail") {
			return "fail";
		}

		JSONObject resultJSON = new JSONObject();
		resultJSON.put("count", 2);

		JSONObject data = new JSONObject();
		// data.put("UserId", (String)obj.get("UserId"));
		data.put("Message", "회원 탈퇴 완료");

		resultJSON.put("data", data);

		return resultJSON.toJSONString();
	}
	
	
	//고지서 부분 수정
		@RequestMapping("/Bill/updateBillField")
		public @ResponseBody JSONObject updateBillField() {
			JSONArray jsonArr = (JSONArray)jsonObject.get("UpdateBillField");
			JSONObject obj = (JSONObject) jsonArr.get(0);
			
			String result = billService.updateBillField(
					(String)obj.get("UserId"),
					(String)obj.get("Date"),
					(String)obj.get("Field"),
					Integer.parseInt(String.valueOf(obj.get("Fee"))));
			
			JSONObject resultJSON = new JSONObject();
			JSONObject data = new JSONObject();
			
			if(result=="fail") {
				data.put("Message", "수정에 실패하였습니다.");
				return resultJSON;
			}
			

			data.put("Message", "수정이 완료되었습니다.");
			data.put("count", 2);
			
			resultJSON.put("Data", data);
			
			return resultJSON;
		}
		
		//고지서 전체 수정
		@PostMapping("/test/update")
		public JSONObject updateBill(@RequestBody TotalBillDTO totalDTO) {
			
			String result = billService.updateBill(totalDTO);
			
			JSONObject resultJSON = new JSONObject();
			resultJSON.put("result", result);
			if(result=="fail") {
				resultJSON.put("message", "고지서 수정 실패하였습니다.");
				return resultJSON;
			}
			
			String UserId = totalDTO.getUserId();
			String date = totalDTO.getDate();
			
			resultJSON.put("message", UserId+"님의 "+date+"의 고지서 수정이 완료되었습니다.");
			
			return resultJSON;
		}
		
		
		//임시로 URL로 설정. 나중에 바꿀예정!
		@RequestMapping("deleteBill")
		public @ResponseBody String deleteBill() {
			JSONArray jsonArr = (JSONArray)jsonObject.get("DeleteBill");
			JSONObject obj = (JSONObject) jsonArr.get(0);
			
			String result = billService.deleteBill((String)obj.get("UserId"),(String)obj.get("Date"));
			return result +":delete";
		}
}
