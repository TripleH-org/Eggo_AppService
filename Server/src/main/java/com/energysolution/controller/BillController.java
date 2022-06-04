package com.energysolution.controller;

import java.util.List;

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

import com.energysolution.dto.BillDTO;
import com.energysolution.dto.DetailBillDTO;
import com.energysolution.dto.PaymentDTO;
import com.energysolution.dto.TotalBillDTO;
import com.energysolution.service.BillServiceImpl;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonInclude.Include;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

@RestController
@RequestMapping("/bill")
public class BillController {
	
	@Autowired
	BillServiceImpl billService;
	
	@GetMapping("")
	public @ResponseBody String mainView() {
		return "This is Billhome";
	}
	
	//특정 날짜 고지서 가져오기
	@GetMapping(value = "/getData", produces = "text/plain;charset=UTF-8")
	public @ResponseBody String getBill(@RequestParam("userId") String UserId, @RequestParam("date") String date) {
		System.out.println("=====특정 날짜 데이터 조회=====");
		System.out.println("userId:"+UserId);
		System.out.println("date:"+date);
		
		TotalBillDTO totalBillDTO = billService.getBill(UserId, date);
			
		Gson gson = new Gson();
		JsonObject resultJSON = new JsonObject();
		
		if(totalBillDTO.getBillId()==-1) {
			System.out.println("fail");
			resultJSON.addProperty("message", UserId+"님 고지서 조회 실패");
			resultJSON.addProperty("result", "fail");
			return gson.toJson(resultJSON);
		}
		
		System.out.println("success");
		resultJSON.addProperty("message", UserId+"님 고지서 조회 성공");
		resultJSON.addProperty("result", "success");
		resultJSON.addProperty("UserId", UserId);
		resultJSON.add("data", totalBillDTO.revertToJson());

		return gson.toJson(resultJSON);
	}
	
	// 원 그래프 데이터 가져오기
	@GetMapping(value = "/pieGraph", produces = "text/plain;charset=UTF-8")
	public @ResponseBody String getPie(@RequestParam("userId") String UserId) {
		System.out.println("=====원그래프데이터 조회=====");
		Gson gson = new Gson();
		JsonObject resultJSON = new JsonObject();

		TotalBillDTO totalBillDTO = billService.getGraphData(UserId);
		if(totalBillDTO.getBillId()==-1) {
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "데이터가 없습니다.");
			return gson.toJson(resultJSON);
		}
	
		resultJSON.addProperty("result", "success");
		resultJSON.addProperty("message", "데이터 조회 성공");
		System.out.println("==fin==");
		System.out.println(totalBillDTO.revertToJsonPie().toString());
		resultJSON.add("data", totalBillDTO.revertToJsonPie());
				
		return gson.toJson(resultJSON);
	}
	
	// 6개월 고지서 가져오기
	@GetMapping(value = "/getDataList", produces = "text/plain;charset=UTF-8")
	public @ResponseBody String getBillTerm(@RequestParam("userId") String UserId) {
		//,@RequestParam("month") int term
		System.out.println("=====6개월 데이터 조회=====");		
		List<TotalBillDTO> listDTO = billService.getBillTerm(UserId, 6);
		
		Gson gson = new Gson();
		JsonObject resultJSON = new JsonObject();
		
		if(listDTO.size()==0) {
			resultJSON.addProperty("message", UserId+"님 6개월 고지서 조회 실패");
			resultJSON.addProperty("result", "fail");
			return gson.toJson(resultJSON);
		}
		
		resultJSON.addProperty("message", UserId+"님 6개월 고지서 조회 성공");
		resultJSON.addProperty("result", "success");
		JsonArray BillList = new JsonArray();
		for(TotalBillDTO dto : listDTO) {
			BillList.add(dto.revertToJson());
		}
		resultJSON.add("dataList", BillList);
		
		return gson.toJson(resultJSON);
	}
	
	//고지서 등록하기
	@PostMapping(value = "/register", produces = "text/plain;charset=UTF-8")
	public @ResponseBody String setBill(TotalBillDTO totalDTO) throws Exception{
		System.out.println("======고지서 등록======");
		Gson gson = new Gson();
		JsonObject resultJSON = new JsonObject();
		if(totalDTO.getDate().equals("")){
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "등록 실패하였습니다.");
			return gson.toJson(resultJSON);
		}
			
		BillDTO billDTO = new BillDTO(totalDTO);
		DetailBillDTO detailbillDTO = new DetailBillDTO(totalDTO);
		PaymentDTO paymentDTO = new PaymentDTO(totalDTO.getUserId());
		String result = billService.insertBill(billDTO,detailbillDTO,paymentDTO);
		
		if(result=="fail") {
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "등록 실패하였습니다.");
			return gson.toJson(resultJSON);
		}
	
		resultJSON.addProperty("result", "success");
		resultJSON.addProperty("message", "등록이 완료되었습니다.");
					
		return gson.toJson(resultJSON);
	}
	
	//고지서 전체 수정
	@PostMapping("/update")
	public @ResponseBody String updateBill(TotalBillDTO totalDTO) {
		System.out.println("======고지서 변경======");
		System.out.println("바뀐고지서==>"+totalDTO.toString());
		
		String result = billService.updateBill(totalDTO);
		
		Gson gson = new Gson();
		JsonObject resultJSON = new JsonObject();
		
		if(result.equals("fail")) {
			resultJSON.addProperty("result", "fail");
			resultJSON.addProperty("message", "고지서 수정 실패하였습니다.");
			return gson.toJson(resultJSON);
		}
		
		String UserId = totalDTO.getUserId();
		String date = totalDTO.getDate();
		
		resultJSON.addProperty("result", "success");
		resultJSON.addProperty("message", UserId+"님의 "+date.substring(0,4)+"년"+date.substring(4)+"월의 고지서 수정이 완료되었습니다.");
		
		return gson.toJson(resultJSON);
	}
	
	
}