package com.energysolution.mapper;

import java.util.HashMap;

import org.apache.ibatis.annotations.Mapper;

import com.energysolution.dto.BillDTO;
import com.energysolution.dto.DetailBillDTO;
import com.energysolution.dto.PaymentDTO;
import com.energysolution.dto.TotalBillDTO;

@Mapper
public interface BillMapper {	
	//등록
	public void insertBill(BillDTO billDTO);
	public void insertDetailBill(DetailBillDTO detailbillDTO);
	public void insertPayment(PaymentDTO paymentDTO);
	
	//조회
	public BillDTO getBill(int BillId);
	public DetailBillDTO getDetailBill(int BillId);
	
	//가장 최근 Data BillId 가져옴
	public int getRecentBillId(String UserId);
	
	//가장 큰 BillId 가져옴
	public int getMaxBillId();
	
	//ID로 조회
	public int getBillId(HashMap<String, String> getBillIdMap);
	public int getBillIdCnt(HashMap<String, String> getBillIdMap);
	
	//변경
	public void updateBillField(HashMap<String, Object> updateBillFieldMap);
	public void updateBill(TotalBillDTO dto);
	public void updateDetailBill(TotalBillDTO dto);
	
	//삭제
	public void deleteBill(int BillId);
	public void deleteDetailBill(int BillId);
	public void deletePayment(int BillId);
	
	//변경확인
	public int checkUpdateBill(int billId);
	public int checkUpdateDetailBill(int billId);
	public int checkUpdateBillField(HashMap<String, Object> updateCheckMap);
	
	//삭제확인
	public int checkDeleteBill(int billId);
	public int checkDeleteDetailBill(int billId);
	public int checkDeletePayment(int billId);
	
	//고지서 확인
	public int checkBill(int BillId);
	
	//payment 고지서 유무 확인
	public int checkPaymentCnt(String UserId);
	
	//UserId, Month에 해당하는 고지서 유무 확인
	public int checkBillDataCnt(HashMap<String, String> map);
}
