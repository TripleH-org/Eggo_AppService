package com.energysolution.service;

import java.util.HashMap;
import java.util.List;

import com.energysolution.dto.BillDTO;
import com.energysolution.dto.DetailBillDTO;
import com.energysolution.dto.PaymentDTO;
import com.energysolution.dto.TotalBillDTO;

public interface BillService {
	//고지서 등록
	public String insertBill(BillDTO billDTO, DetailBillDTO detailbillDTO,PaymentDTO paymentDTO);
	
	// 그래프용 데이터
	public TotalBillDTO getGraphData(String UserId);
	
	// 특정 기간 고지서 조회 (UserId와 Term)
	public List<TotalBillDTO> getBillTerm(String UserId, int term);
	
	// 특정 날짜 고지서 조회 (UserId와 Term)
	public TotalBillDTO getBill(String UserId, String date);
	
	// 고지서 부분 수정
	public String updateBillField(String UserId, String Date, String Field, int fee);
	
	// 고지서 전체 수정
	public String updateBill(TotalBillDTO totalDTO);
	
	// 고지서 삭제
	public String deleteBill(String UserId, String Date);
	
	// 고지서 유무 확인
	public String checkBill(int billId);
	
	// 고지서 특정 필드 값 변경 확인
	public String checkUpdateBillField(HashMap<String, Object> updateCheckBillMap);

	// 고지서 전체 필드 값 변경 확인
	public String checkUpdateBill(int BillId, HashMap<String, Object> updateBillMap);
		
	// 고지서 삭제 확인
	public String checkDeleteBill(int billId);
	
	// 고지서 ID 가져오기
	public int getBillId(String UserId, String date);

	int getBillIdCnt(String UserId, String date);


}
