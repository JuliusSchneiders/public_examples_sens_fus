#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstração de Fusão Sensorial (Câmera, LiDAR, IMU) - Padrão KITTI Dataset.

Este script simula o pipeline de processamento para fusão de dados em veículos
autônomos. O foco está na sincronização temporal e na projeção geométrica
entre sistemas de coordenadas distintos (Mundo -> Velodyne -> Câmera).

Requisitos:
    - numpy
    - dataclasses (nativo Python 3.7+)
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


@dataclass
class SensorFrame:
    """
    Representa um snapshot síncrono dos dados dos sensores.
    
    Attributes:
        timestamp (float): Tempo do sistema no momento da captura.
        image_data (np.ndarray): Matriz de pixels (H, W, C).
        lidar_points (np.ndarray): Nuvem de pontos (N, 4) -> (x, y, z, refletância).
        imu_data (Dict[str, float]): Dados inerciais (aceleração, giroscópio).
    """
    timestamp: float
    image_data: Optional[np.ndarray] = None
    lidar_points: Optional[np.ndarray] = None
    imu_data: Dict[str, float] = field(default_factory=dict)


class KittiCalibration:
    """
    Gerencia as matrizes de calibração intrínsecas e extrínsecas.
    
    Nota para a turma: No dataset KITTI, a 'verdade' geométrica depende
    da multiplicação correta dessas matrizes. A ordem importa!
    """
    
    def __init__(self) -> None:
        # Matriz de Projeção da Câmera (Intrínseca) - P_rect_xx
        # Simulação de uma câmera com distância focal f=707px
        self.P_rect: np.ndarray = np.array([
            [7.07e2, 0.0, 6.09e2, 0.0],
            [0.0, 7.07e2, 1.80e2, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

        # Matriz de Rotação Retificadora (Stereo) - R_rect_xx
        self.R_rect: np.ndarray = np.eye(4)  # Identidade 4x4 para simplificação

        # Matriz Extrínseca: Velodyne (LiDAR) para Câmera (Referência)
        # Tr_velo_to_cam
        self.Tr_velo_to_cam: np.ndarray = np.array([
            [0.0, -1.0, 0.0, 0.0],  # Exemplo de rotação 90 graus + translação
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, -0.08], # Lidar está 8cm atrás da câmera (exemplo)
            [0.0, 0.0, 0.0, 1.0]
        ])

    def project_lidar_to_image(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Projeta pontos 3D do sistema de coordenadas do LiDAR para o plano 2D da imagem.
        
        Equação Fundamental:
            y_image = P_rect * R_rect * Tr_velo_to_cam * x_lidar
        
        Args:
            points_3d (np.ndarray): Pontos (N, 3) ou (N, 4).
        
        Returns:
            np.ndarray: Coordenadas (u, v) na imagem.
        """
        num_points = points_3d.shape[0]
        
        # 1. Converter para coordenadas homogêneas (adicionar 1 na 4ª dimensão)
        if points_3d.shape[1] == 3:
            points_h = np.hstack((points_3d, np.ones((num_points, 1))))
        else:
            points_h = points_3d
            
        # 2. Transformação Rígida: LiDAR -> Câmera Ref
        points_cam_ref = self.Tr_velo_to_cam @ points_h.T
        
        # 3. Retificação (Alinhamento Stereo)
        points_rect = self.R_rect @ points_cam_ref
        
        # 4. Projeção Perspectiva (3D -> 2D)
        # Note que P_rect é 3x4, resultando em vetores 3xN
        projected_h = self.P_rect @ points_rect  
        
        # 5. Normalização Homogênea (dividir x e y por z)
        # u = x / z, v = y / z
        uv_coords = projected_h[:2, :] / projected_h[2, :]
        
        return uv_coords.T


class SensorFusionEngine:
    """
    Controlador principal do pipeline de fusão.
    """
    
    def __init__(self, use_synthetic_data: bool = True) -> None:
        self.calibration = KittiCalibration()
        self.use_synthetic = use_synthetic_data
        print("[SISTEMA] Engine de Fusão Inicializada.")

    def _generate_synthetic_lidar(self, num_points: int = 100) -> np.ndarray:
        """Gera pontos aleatórios à frente do veículo para teste."""
        # X: Frente (0 a 50m), Y: Esquerda/Direita (-10 a 10m), Z: Altura (-2 a 2m)
        x = np.random.uniform(5, 50, num_points)
        y = np.random.uniform(-10, 10, num_points)
        z = np.random.uniform(-2, 1, num_points)
        reflectance = np.random.uniform(0, 1, num_points)
        return np.column_stack((x, y, z, reflectance))

    def _get_synced_frame(self) -> SensorFrame:
        """
        Simula a etapa crítica de 'Soft Synchronization'.
        Em um cenário real, buscaríamos timestamps próximos com tolerância (ex: < 10ms).
        """
        now = time.time()
        
        # Simulando aquisição de dados
        lidar = self._generate_synthetic_lidar(500)
        img = np.zeros((375, 1242, 3), dtype=np.uint8) # Tamanho padrão KITTI
        imu = {"ax": 0.01, "ay": -0.04, "az": 9.81} # Aceleração quase estática
        
        return SensorFrame(
            timestamp=now,
            image_data=img,
            lidar_points=lidar,
            imu_data=imu
        )

    def run_pipeline(self, steps: int = 5) -> None:
        """Executa o loop principal de processamento."""
        print(f"\n[PIPELINE] Iniciando processamento de {steps} frames...")
        
        for i in range(steps):
            frame = self._get_synced_frame()
            
            # Passo 1: Filtragem (ROI - Region of Interest)
            # Removemos pontos atrás do veículo (x < 0 no frame do Velodyne)
            # O eixo X do Velodyne aponta para frente.
            valid_indices = frame.lidar_points[:, 0] > 0
            points_frontal = frame.lidar_points[valid_indices]
            
            # Passo 2: Fusão Geométrica (Projeção)
            # Projetamos os pontos 3D na imagem 2D
            uv_coords = self.calibration.project_lidar_to_image(points_frontal)
            
            # Passo 3: Validação (Check se está dentro do canvas da imagem)
            h, w, _ = frame.image_data.shape
            in_view = (uv_coords[:, 0] >= 0) & (uv_coords[:, 0] < w) & \
                      (uv_coords[:, 1] >= 0) & (uv_coords[:, 1] < h)
            
            points_in_image = uv_coords[in_view]
            
            print(f"--- Frame {i+1} ---")
            print(f"Timestamp IMU/LIDAR: {frame.timestamp:.4f}")
            print(f"Pontos LiDAR Brutos: {len(frame.lidar_points)}")
            print(f"Pontos Projetados Válidos (Fusão Visual): {len(points_in_image)}")
            print(f"Exemplo de Projeção (u, v): {points_in_image[0].astype(int) if len(points_in_image) > 0 else 'N/A'}")
            
            # Aqui poderíamos usar cv2.circle para desenhar na imagem
            # frame.image_data = draw_lidar_on_image(...)


if __name__ == "__main__":
    print("="*60)
    print("PERCEPÇÃO COMPUTACIONAL")
    print("Arquitetura de Fusão de Sensores (KITTI/Waymo specs)")
    print("="*60)
    
    # Instanciação e Execução
    fusion_system = SensorFusionEngine(use_synthetic_data=True)
    fusion_system.run_pipeline(steps=3)
    
    print("\n[CONCLUSÃO] Pipeline finalizado. Verifique os logs de projeção.")
